import os, logging, tempfile, shutil
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

# Create a secure directory to host the generated 3D meshes
os.makedirs("static/models", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

HF_TOKENS = [os.getenv(f"HF_TOKEN_{i}", "").strip() for i in range(1, 5) if os.getenv(f"HF_TOKEN_{i}")]

@app.post("/api/v1/convert-2d-to-3d")
async def convert_2d_to_3d(image: UploadFile = File(...)):
    """
    Accepts 2D images, processes via Hugging Face TripoSR Space, 
    and returns a live hosted URL of the generated 3D .glb mesh.
    """
    try:
        # 1. Securely cache incoming image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            content = await image.read()
            tmp_img.write(content)
            img_path = tmp_img.name

        # 2. Establish neural connection
        token = HF_TOKENS[0] if HF_TOKENS else None
        logger.info("[engine] Dispatching asset to neural mesh constructor...")
        client = Client("stabilityai/TripoSR", token=token)

        # 3. Execute 3D mathematical generation# 3. Execute 3D mathematical generation
        result = client.predict(
            handle_file(img_path),
            True,    # do_remove_background
            0.85,    # foreground_ratio
            api_name="/generate",
        )

        glb_src = result[1] if isinstance(result, (list, tuple)) else result

        # 4. Relocate the generated mesh to the public hosting directory
        filename = f"mesh_{os.path.basename(glb_src)}"
        if not filename.endswith('.glb'):
            filename += '.glb'
            
        dest_path = os.path.join("static", "models", filename)
        shutil.copy(glb_src, dest_path)

        # 5. Terminate temporary artifacts
        os.unlink(img_path)

        return {
            "success": True,
            "model_url": f"https://jewelry-3d-api.onrender.com/{dest_path}"
        }

    except Exception as e:
        logger.error(f"[engine] Core failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
