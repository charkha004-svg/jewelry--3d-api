"""
2D-to-3D Reconstructor Engine — Fixed Version v3
Fixes: correct /generate args (image + resolution), handle_file on preprocessed path
"""
import os, logging, tempfile, shutil, traceback, uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from gradio_client import Client, handle_file
from dotenv import load_dotenv

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


def extract_glb_path(result) -> str:
    """
    /generate returns [obj_filedata, glb_filedata]
    Each filedata is a dict with a 'path' key, or a plain string path.
    We want index 1 (GLB).
    """
    logger.info(f"[engine] Raw result type: {type(result)}, value: {repr(result)[:400]}")

    # Expected: list/tuple of two items, we want index 1 (GLB)
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        candidate = result[1]
    elif isinstance(result, (list, tuple)) and len(result) == 1:
        candidate = result[0]
    else:
        candidate = result

    # Gradio FileData dict
    if isinstance(candidate, dict):
        for key in ["path", "name", "url"]:
            if candidate.get(key):
                return candidate[key]

    # Plain string path
    if isinstance(candidate, str) and candidate:
        return candidate

    raise ValueError(f"Cannot extract GLB path from result: {repr(result)[:300]}")


@app.get("/health")
async def health():
    return {"status": "ok", "tokens_loaded": len(HF_TOKENS)}


@app.post("/api/v1/convert-2d-to-3d")
async def convert_2d_to_3d(image: UploadFile = File(...)):
    img_path = None
    try:
        # 1. Save to temp file
        suffix = os.path.splitext(image.filename or "item.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await image.read()
            tmp.write(content)
            img_path = tmp.name
        logger.info(f"[engine] Image saved: {img_path} ({len(content)} bytes)")

        # 2. Token + fresh client
        token = get_next_token()
        logger.info("[engine] Connecting to TripoSR...")
        client = Client("stabilityai/TripoSR", token=token)

        # 3. STEP A — preprocess (background removal)
        # Args: (image, do_remove_background: bool, foreground_ratio: float)
        logger.info("[engine] Running /preprocess...")
        preprocessed = client.predict(
            handle_file(img_path),
            True,   # remove background
            0.85,   # foreground ratio
            api_name="/preprocess",
        )
        logger.info(f"[engine] Preprocessed path: {repr(preprocessed)[:200]}")

        # 4. STEP B — generate 3D
        # Args: (processed_image, marching_cubes_resolution: float 32-320)
        # Returns: [obj_filedata, glb_filedata]
        logger.info("[engine] Running /generate...")

        # preprocessed is a filepath string — wrap it for the next call
        if isinstance(preprocessed, str):
            processed_input = handle_file(preprocessed)
        elif isinstance(preprocessed, dict) and "path" in preprocessed:
            processed_input = handle_file(preprocessed["path"])
        else:
            processed_input = preprocessed

        result = client.predict(
            processed_input,
            256,    # marching cubes resolution (higher = more detail, slower)
            api_name="/generate",
        )

        # 5. Extract GLB path
        glb_src = extract_glb_path(result)
        logger.info(f"[engine] GLB source: {glb_src}")

        if not os.path.exists(glb_src):
            raise FileNotFoundError(f"GLB not found at: {glb_src}")

        # 6. Copy to public dir
        filename = f"mesh_{uuid.uuid4().hex[:8]}.glb"
        dest_path = os.path.join("static", "models", filename)
        shutil.copy(glb_src, dest_path)
        logger.info(f"[engine] Mesh hosted at: {dest_path}")

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
        if img_path and os.path.exists(img_path):
            try:
                os.unlink(img_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
