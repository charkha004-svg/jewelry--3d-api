"""
2D-to-3D Reconstructor Engine — Fixed Version
Fixes: TypeError from result parsing, stale HF client, better fallback
"""
import os, logging, tempfile, shutil, traceback
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
    TripoSR returns results in multiple formats depending on Space version.
    This handles all known return shapes safely.
    """
    logger.info(f"[engine] Raw result type: {type(result)}, value: {repr(result)[:300]}")

    # Case 1: list or tuple — try index 1 first (mesh), then 0
    if isinstance(result, (list, tuple)):
        for idx in [1, 0]:
            if idx < len(result) and result[idx]:
                candidate = result[idx]
                # Sometimes it's a dict with a 'name' key (gradio FileData)
                if isinstance(candidate, dict) and "name" in candidate:
                    return candidate["name"]
                if isinstance(candidate, str) and (candidate.endswith(".glb") or candidate.endswith(".obj")):
                    return candidate
                # Could be a path in a sub-list
                if isinstance(candidate, (list, tuple)) and len(candidate) > 0:
                    sub = candidate[0]
                    if isinstance(sub, dict) and "name" in sub:
                        return sub["name"]
                    if isinstance(sub, str):
                        return sub

    # Case 2: dict directly
    if isinstance(result, dict):
        for key in ["name", "path", "file", "output"]:
            if key in result and result[key]:
                return result[key]

    # Case 3: string path
    if isinstance(result, str) and result:
        return result

    raise ValueError(f"Cannot extract GLB path from result: {repr(result)[:200]}")


@app.get("/health")
async def health():
    return {"status": "ok", "tokens_loaded": len(HF_TOKENS)}


@app.post("/api/v1/convert-2d-to-3d")
async def convert_2d_to_3d(image: UploadFile = File(...)):
    """
    Accepts a 2D jewelry image, runs TripoSR on HuggingFace,
    hosts the .glb, and returns a live URL.
    """
    img_path = None
    try:
        # 1. Save incoming image to temp file
        suffix = os.path.splitext(image.filename or "item.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await image.read()
            tmp.write(content)
            img_path = tmp.name

        logger.info(f"[engine] Image saved to {img_path} ({len(content)} bytes)")

        # 2. Get token — rotate to avoid quota hits
        token = get_next_token()
        if not token:
            logger.warning("[engine] No HF token available, trying without auth")

        # 3. Fresh client per request (avoids stale session / 404 heartbeat issue)
        logger.info("[engine] Connecting to TripoSR space...")
        client = Client("stabilityai/TripoSR", hf_token=token)

        # 4. Run prediction
        logger.info("[engine] Running /preprocess then /generate...")

        # Step A: preprocess (remove background)
        preprocessed = client.predict(
            handle_file(img_path),
            True,    # do_remove_background
            0.85,    # foreground_ratio
            api_name="/preprocess",
        )
        logger.info(f"[engine] Preprocess result: {repr(preprocessed)[:200]}")

        # Step B: generate 3D mesh
        result = client.predict(
            preprocessed,
            api_name="/generate",
        )

        # 5. Safely extract .glb file path
        glb_src = extract_glb_path(result)
        logger.info(f"[engine] GLB source path: {glb_src}")

        if not os.path.exists(glb_src):
            raise FileNotFoundError(f"GLB file not found at: {glb_src}")

        # 6. Copy to public hosting dir
        import uuid
        filename = f"mesh_{uuid.uuid4().hex[:8]}.glb"
        dest_path = os.path.join("static", "models", filename)
        shutil.copy(glb_src, dest_path)
        logger.info(f"[engine] Mesh saved to {dest_path}")

        # Render's base URL from env (set this in Render dashboard)
        base_url = os.getenv("RENDER_EXTERNAL_URL", "https://jewelry-3d-api.onrender.com")
        model_url = f"{base_url}/static/models/{filename}"

        return {
            "success": True,
            "model_url": model_url,
            "filename": filename,
        }

    except Exception as e:
        logger.error(f"[engine] Core failure: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up temp file
        if img_path and os.path.exists(img_path):
            try:
                os.unlink(img_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
