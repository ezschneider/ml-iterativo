from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import uuid
import os
from app.ml.pipeline import run_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_dataset(file: UploadFile, target_column: str = Form(...)):
    try:
        contents = await file.read()
        job_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{job_id}.csv")

        with open(file_path, "wb") as f:
            f.write(contents)

        df = pd.read_csv(file_path)

        if target_column not in df.columns:
            return JSONResponse(status_code=400, content={"error": "Target column not found in dataset."})

        result = run_pipeline(df, target_column)

        return {"job_id": job_id, "result": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})