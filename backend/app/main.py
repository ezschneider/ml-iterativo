from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import pandas as pd
import uuid
from io import BytesIO
from app.ml.pipeline import MLPipeline

app = FastAPI()

# Middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para limitar tamanho do upload
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

class FileSizeLimiterMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_FILE_SIZE:
            return JSONResponse(status_code=413, content={"error": "Arquivo excede o limite de 5MB."})
        return await call_next(request)

app.add_middleware(FileSizeLimiterMiddleware)

@app.post("/upload")
async def upload_dataset(file: UploadFile, target_column: str = Form(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        if target_column not in df.columns:
            return JSONResponse(status_code=400, content={"error": "Target column not found in dataset."})

        job_id = str(uuid.uuid4())
        pipeline = MLPipeline(df, target_column)
        result = pipeline.run()

        return {"job_id": job_id, "result": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})