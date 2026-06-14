"""
2D-to-3D Reconstructor Engine — Render deployment (main.py)
THE FIX: Returns model_url pointing to THIS server's /static/models/
so Flask can pass it directly to model-viewer without re-downloading.
"""
import os, logging, tempfile, traceback, uuid, time, requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="2D-to-3D Reconstructor Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main_3d")

os.makedirs("static/models", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

TRIPO_API_KEY    = os.getenv("TRIPO_API_KEY", "")
TRIPO_BASE       = "https://api.tripo3d.ai/v2/openapi"

# ── IMPORTANT: Set this in Render dashboard env vars ─────────────────────────
# RENDER_EXTERNAL_URL = https://jewelry-3d-api.onrender.com   (no trailing slash)
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "https://jewelry-3d-api.onrender.com")

@app.get("/health")
async def health():
    return {"status": "ok", "engine": "tripo3d", "key_set": bool(TRIPO_API_KEY)}

@app.post("/api/v1/convert-2d-to-3d")
async def convert_2d_to_3d(file: UploadFile = File(...)):
    tmp_path = None
    try:
        # 1. Save image temporarily
        suffix = os.path.splitext(file.filename or "item.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        logger.info(f"[engine] Image saved: {tmp_path} ({len(content)} bytes)")

        headers = {"Authorization": f"Bearer {TRIPO_API_KEY}"}

        # 2. Upload to Tripo
        logger.info("[engine] Uploading image to Tripo3D...")
        with open(tmp_path, "rb") as f:
            upload_resp = requests.post(
                f"{TRIPO_BASE}/upload",
                headers=headers,
                files={"file": (file.filename or "item.jpg", f, "image/jpeg")},
                timeout=30,
            )
        upload_resp.raise_for_status()
        image_token = upload_resp.json()["data"]["image_token"]
        logger.info(f"[engine] Image token: {image_token}")

        # 3. Submit task
        logger.info("[engine] Submitting image-to-3D task...")
        task_resp = requests.post(
            f"{TRIPO_BASE}/task",
            headers={**headers, "Content-Type": "application/json"},
            json={"type": "image_to_model", "file": {"type": "jpg", "file_token": image_token}},
            timeout=30,
        )
        task_resp.raise_for_status()
        task_id = task_resp.json()["data"]["task_id"]
        logger.info(f"[engine] Task ID: {task_id}")

        # 4. Poll Tripo (max 5 min)
        logger.info("[engine] Polling for task completion...")
        for _ in range(60):
            time.sleep(5)
            status_resp = requests.get(f"{TRIPO_BASE}/task/{task_id}", headers=headers, timeout=15)
            status_resp.raise_for_status()
            task_data = status_resp.json()["data"]
            status    = task_data["status"]
            progress  = task_data.get("progress", 0)
            logger.info(f"[engine] Status: {status} | Progress: {progress}%")

            if status == "success":
                glb_url = task_data["output"].get("pbr_model") or task_data["output"].get("model")
                logger.info(f"[engine] GLB URL: {glb_url}")
                break
            elif status in ("failed", "cancelled", "unknown"):
                raise RuntimeError(f"Tripo task {status}")
        else:
            raise TimeoutError("Tripo3D timed out after 5 minutes")

        # 5. Download GLB and save locally on THIS Render server
        logger.info("[engine] Downloading GLB...")
        glb_resp = requests.get(glb_url, timeout=120)
        glb_resp.raise_for_status()

        filename  = f"mesh_{uuid.uuid4().hex[:8]}.glb"
        dest_path = os.path.join("static", "models", filename)
        with open(dest_path, "wb") as f:
            f.write(glb_resp.content)
        logger.info(f"[engine] Mesh saved: {dest_path} ({len(glb_resp.content)} bytes)")

        # ✅ Return THIS SERVER'S URL — not the expiring Tripo signed URL
        final_url = f"{RENDER_EXTERNAL_URL}/static/models/{filename}"
        logger.info(f"[engine] Returning model_url: {final_url}")

        return {
            "success":   True,
            "model_url": final_url,   # e.g. https://jewelry-3d-api.onrender.com/static/models/mesh_abc.glb
            "filename":  filename,
            "task_id":   task_id,
            "fallback":  False,
        }

    except Exception as e:
        logger.error(f"[engine] Core failure: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.unlink(tmp_path)
            except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
