"""
2D-to-3D Reconstructor Engine — v4 with image quality enhancement
Fixes: better preprocessing, max resolution, image sharpening before TripoSR
"""
import os, logging, tempfile, shutil, traceback, uuid, io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from gradio_client import Client, handle_file
from dotenv import load_dotenv
from PIL import Image, ImageFilter, ImageEnhance

load_dotenv()

app = FastAPI(title="2D-to-3D Reconstructor Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs("static/models", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

HF_TOKENS = [os.getenv(f"HF_TOKEN_{i}", "").strip() for i in range(1, 5) if os.getenv(f"HF_TOKEN_{i}")]
_token_idx = 0

def get_next_token():
    global _token_idx
    if not HF_TOKENS:
        return None
    t = HF_TOKENS[_token_idx % len(HF_TOKENS)]
    _token_idx += 1
    return t


def enhance_image_for_3d(input_path: str) -> str:
    """
    Enhance image quality before sending to TripoSR:
    - Resize to optimal 512x512 (TripoSR's sweet spot)
    - Sharpen + boost contrast so details are crisp
    - Place on pure white background (helps background removal)
    - Save as high-quality PNG (not JPEG — no compression artifacts)
    """
    img = Image.open(input_path).convert("RGBA")
    
    # Resize to 512x512 with padding on white background
    img.thumbnail((512, 512), Image.LANCZOS)
    canvas = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
    offset = ((512 - img.width) // 2, (512 - img.height) // 2)
    canvas.paste(img, offset, img if img.mode == "RGBA" else None)
    
    # Convert to RGB, sharpen and boost contrast
    rgb = canvas.convert("RGB")
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    rgb = ImageEnhance.Contrast(rgb).enhance(1.3)
    rgb = ImageEnhance.Sharpness(rgb).enhance(2.0)
    
    # Save as PNG (lossless — much better for TripoSR)
    enhanced_path = input_path.replace(".jpg", "_enhanced.png").replace(".jpeg", "_enhanced.png")
    if not enhanced_path.endswith(".png"):
        enhanced_path += "_enhanced.png"
    rgb.save(enhanced_path, format="PNG")
    logger.info(f"[engine] Enhanced image saved: {enhanced_path} ({rgb.size})")
    return enhanced_path


def extract_glb_path(result) -> str:
    logger.info(f"[engine] Raw result type: {type(result)}, value: {repr(result)[:400]}")
    # /generate returns (obj_filedata, glb_filedata) — we want index 1
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        candidate = result[1]
    elif isinstance(result, (list, tuple)) and len(result) == 1:
        candidate = result[0]
    else:
        candidate = result

    if isinstance(candidate, dict):
        for key in ["path", "name", "url"]:
            if candidate.get(key):
                return candidate[key]
    if isinstance(candidate, str) and candidate:
        return candidate

    raise ValueError(f"Cannot extract GLB path from result: {repr(result)[:300]}")


@app.get("/health")
async def health():
    return {"status": "ok", "tokens_loaded": len(HF_TOKENS)}


@app.post("/api/v1/convert-2d-to-3d")
async def convert_2d_to_3d(image: UploadFile = File(...)):
    img_path = None
    enhanced_path = None
    try:
        # 1. Save incoming image
        suffix = os.path.splitext(image.filename or "item.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await image.read()
            tmp.write(content)
            img_path = tmp.name
        logger.info(f"[engine] Image saved: {img_path} ({len(content)} bytes)")

        # 2. Enhance image quality before TripoSR
        enhanced_path = enhance_image_for_3d(img_path)

        # 3. Fresh client
        token = get_next_token()
        logger.info("[engine] Connecting to TripoSR...")
        client = Client("stabilityai/TripoSR", token=token)

        # 4. STEP A — preprocess with background removal
        # Use foreground_ratio=0.90 for jewelry (fills more of the frame)
        logger.info("[engine] Running /preprocess...")
        preprocessed = client.predict(
            handle_file(enhanced_path),
            True,   # remove background
            0.90,   # foreground ratio — higher = jewelry fills more frame
            api_name="/preprocess",
        )
        logger.info(f"[engine] Preprocessed: {repr(preprocessed)[:200]}")

        # 5. STEP B — generate at MAX resolution (320)
        logger.info("[engine] Running /generate at max resolution (320)...")
        if isinstance(preprocessed, str):
            processed_input = handle_file(preprocessed)
        elif isinstance(preprocessed, dict) and "path" in preprocessed:
            processed_input = handle_file(preprocessed["path"])
        else:
            processed_input = preprocessed

        result = client.predict(
            processed_input,
            320,    # MAX marching cubes resolution for best quality
            api_name="/generate",
        )

        # 6. Extract and host GLB
        glb_src = extract_glb_path(result)
        logger.info(f"[engine] GLB source: {glb_src}")

        if not os.path.exists(glb_src):
            raise FileNotFoundError(f"GLB not found at: {glb_src}")

        filename = f"mesh_{uuid.uuid4().hex[:8]}.glb"
        dest_path = os.path.join("static", "models", filename)
        shutil.copy(glb_src, dest_path)
        logger.info(f"[engine] Mesh hosted: {dest_path}")

        base_url = os.getenv("RENDER_EXTERNAL_URL", "https://jewelry-3d-api.onrender.com")
        return {
            "success": True,
            "model_url": f"{base_url}/static/models/{filename}",
            "filename": filename,
        }

    except Exception as e:
        logger.error(f"[engine] Core failure: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for path in [img_path, enhanced_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
