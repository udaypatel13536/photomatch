from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tempfile
import os
import logging
import urllib.request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PhotoMatch AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEIGHTS_DIR = "/app/weights"
DETECTOR_PATH = os.path.join(WEIGHTS_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_PATH = os.path.join(WEIGHTS_DIR, "face_recognition_sface_2021dec.onnx")

face_detector = None
face_recognizer = None


@app.on_event("startup")
async def load_models():
    global face_detector, face_recognizer
    face_detector = cv2.FaceDetectorYN.create(DETECTOR_PATH, "", (320, 320))
    face_recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_PATH, "")
    logger.info("Models loaded successfully!")


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "SFace", "detector": "yunet"}


@app.post("/extract-embedding")
async def extract_embedding(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    suffix = os.path.splitext(file.filename or "img.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        img = cv2.imread(tmp_path)
        if img is None:
            return {"faces": [], "face_count": 0, "error": "Could not read image"}

        h, w = img.shape[:2]
        face_detector.setInputSize((w, h))
        _, detections = face_detector.detect(img)

        faces = []
        if detections is not None:
            for det in detections:
                confidence = float(det[-1])
                if confidence < 0.5:
                    continue
                aligned = face_recognizer.alignCrop(img, det)
                embedding = face_recognizer.feature(aligned).flatten().tolist()
                x, y, fw, fh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                faces.append({
                    "embedding": embedding,
                    "facial_area": {"x": x, "y": y, "w": fw, "h": fh},
                    "confidence": confidence,
                })

        return {"faces": faces, "face_count": len(faces)}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"faces": [], "face_count": 0, "error": str(e)}

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
