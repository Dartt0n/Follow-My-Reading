from fastapi import APIRouter, UploadFile, HTTPException, status
from uuid import uuid4, UUID
from .models import UploadFileResponse
import aiofiles

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/upload", response_model=UploadFileResponse)
async def upload_audio(upload_file: UploadFile) -> UploadFileResponse:
    file_id = uuid4()

    # Here we using MIME types specification, which have format
    # "kind/name". In the following code, we checking that the kind of document
    # is "audio". It is the easiest methods to allow uploading any audio format files.
    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types
    if (
        upload_file.content_type is None
        or upload_file.content_type.split("/")[0] != "audio"
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only audio files uploads are allowed",
        )

    async with aiofiles.open("./temp_data/audio/" + str(file_id), "wb") as file:
        byte_content = await upload_file.read()
        await file.write(byte_content)

    return UploadFileResponse(file_id=file_id)
