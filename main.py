from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import numpy as np
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PhotoMatch AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def load_model():
    """Pre-load the SFace model on startup so first request is fast."""
    logger.info("Pre-loading SFace model...")
    try:
        DeepFace.represent(
            img_path=np.zeros((224, 224, 3), dtype=np.uint8),
            model_name="SFace",
            detector_backend="skip",
            enforce_detection=False,
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.warning(f"Model pre-load note: {e}")


@app.post("/extract-embedding")
async def extract_embedding(file: UploadFile = File(...)):
    """
    Accept an image, detect all faces, return 128-dim embeddings (SFace).
    Using SFace + opencv to stay within Render free tier 512 MB RAM limit.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    suffix = os.path.splitext(file.filename or "img.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        results = DeepFace.represent(
            img_path=tmp_path,
            model_name="SFace",          # 128-dim, ~400 MB RAM — fits free tier
            detector_backend="opencv",   # lighter than retinaface
            enforce_detection=False,
            align=True,
        )

        faces = []
        for r in results:
            confidence = r.get("face_confidence", 0)
            if confidence > 0.5:
                faces.append({
                    "embedding": r["embedding"],
                    "facial_area": r["facial_area"],
                    "confidence": float(confidence),
                })

        return {"faces": faces, "face_count": len(faces)}

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"faces": [], "face_count": 0, "error": str(e)}

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "SFace", "detector": "opencv"}
