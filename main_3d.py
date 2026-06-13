"""
2D-to-3D Reconstructor Engine — Tripo3D API Version
Professional quality 3D reconstruction for jewelry
"""
import os, logging, tempfile, shutil, traceback, uuid, time, requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

TRIPO_API_KEY = os.getenv("TRIPO_API_KEY", "")
TRIPO_BASE    = "https://api.tripo3d.ai/v2/openapi"


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "tripo3d", "key_set": bool(TRIPO_API_KEY)}


@app.post("/api/v1/convert-2d-to-3d")
async def convert_2d_to_3d(file: UploadFile = File(...)):
    tmp_path = None
    try:
        # 1. Save incoming image
        suffix = os.path.splitext(file.filename or "item.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        logger.info(f"[engine] Image saved: {tmp_path} ({len(content)} bytes)")

        headers = {"Authorization": f"Bearer {TRIPO_API_KEY}"}

        # 2. Upload image to Tripo
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

        
        # 3. Submit image-to-3D task
        logger.info("[engine] Submitting image-to-3D task...")
        task_resp = requests.post(
            f"{TRIPO_BASE}/task",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "type": "image_to_model",
                "file": {
                    "type": "jpg",
                    "file_token": image_token,
                }
            },
            timeout=30,
        )
        task_resp.raise_for_status()
        task_id = task_resp.json()["data"]["task_id"]
        logger.info(f"[engine] Task ID: {task_id}")

        # 4. Poll for completion (max 5 minutes)
        logger.info("[engine] Polling for task completion...")
        for attempt in range(60):  # 60 x 5s = 5 minutes
            time.sleep(5)
            status_resp = requests.get(
                f"{TRIPO_BASE}/task/{task_id}",
                headers=headers,
                timeout=15,
            )
            status_resp.raise_for_status()
            task_data = status_resp.json()["data"]
            status    = task_data["status"]
            progress  = task_data.get("progress", 0)
            logger.info(f"[engine] Status: {status} | Progress: {progress}%")

            if status == "success":
                glb_url = task_data["output"]["pbr_model"] or task_data["output"]["model"]
                logger.info(f"[engine] GLB URL: {glb_url}")
                break
            elif status in ("failed", "cancelled", "unknown"):
                raise RuntimeError(f"Tripo task {status}: {task_data.get('message', '')}")
        else:
            raise TimeoutError("Tripo3D task timed out after 5 minutes")

        # 5. Download the GLB
        logger.info("[engine] Downloading GLB...")
        glb_resp = requests.get(glb_url, timeout=60)
        glb_resp.raise_for_status()

        filename  = f"mesh_{uuid.uuid4().hex[:8]}.glb"
        dest_path = os.path.join("static", "models", filename)
        with open(dest_path, "wb") as f:
            f.write(glb_resp.content)
        logger.info(f"[engine] Mesh saved: {dest_path} ({len(glb_resp.content)} bytes)")

        base_url = os.getenv("RENDER_EXTERNAL_URL", "https://jewelry-3d-api.onrender.com")
        return {
            "success":   True,
            "model_url": f"{base_url}/static/models/{filename}",
            "filename":  filename,
            "task_id":   task_id,
        }

    except Exception as e:
        logger.error(f"[engine] Core failure: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
