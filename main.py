from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Annotated
from uuid import uuid4, UUID
from datetime import datetime
import pytesseract
import whisper

app = FastAPI()

model = whisper.load_model("base")


class Response(BaseModel):
    type: str
    size: int
    uuid: UUID
    extracted: object


@app.post("/upload/image", response_model=Response)
async def upload_image(file: UploadFile):
    """
    Convert image to text
    """
    byte_data = await file.read()
    file_uuid = str(uuid4())

    with open("./temp/" + file_uuid, "wb") as f:
        f.write(byte_data)

    result = pytesseract.image_to_string(file_uuid, lang="eng")

    # result = model.transcribe("./temp/" + file_uuid)

    return {
        "type": file.content_type,
        "size": file.size,
        "uuid": file_uuid,
        "extracted": result,
    }


@app.post("/upload/audio")
async def upload_audio(file: UploadFile):
    start = datetime.now()
    byte_data = await file.read()
    file_uuid = str(uuid4())

    with open("./temp/" + file_uuid, "wb") as f:
        f.write(byte_data)

    # result = pytesseract.image_to_string(file_uuid, lang="eng")

    result = model.transcribe("./temp/" + file_uuid)

    end = datetime.now()
    return {
        "type": file.content_type,
        "size": file.size,
        "uuid": file_uuid,
        "extracted": result,
        "took": (end - start).microseconds,
    }
