from uuid import uuid4, UUID

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from pathlib import Path
from pydantic.error_wrappers import ValidationError

from core.plugins.no_mem import get_image_plugins
from .task import _get_job_status, _get_job_result, create_image_task
from .auth import get_current_active_user
from .models import (
    ImageProcessingRequest,
    ImageProcessingResponse,
    ModelData,
    ModelsDataReponse,
    UploadFileResponse,
    TaskCreateResponse,
)

router = APIRouter(
    prefix="/image", tags=["image"], dependencies=[Depends(get_current_active_user)]
)


@router.post("/upload", response_model=UploadFileResponse)
async def upload_image(upload_file: UploadFile) -> UploadFileResponse:
    file_id = uuid4()

    # Here we using MIME types specification, which have format
    # "kind/name". In the following code, we checking that the kind of document
    # is "image". It is the easiest methods to allow uploading any image format files.
    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types
    if (
        upload_file.content_type is None
        or upload_file.content_type.split("/")[0] != "image"
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only image files uploads are allowed",
        )

    async with aiofiles.open("./temp_data/image/" + str(file_id), "wb") as file:
        byte_content = await upload_file.read()
        await file.write(byte_content)
    return UploadFileResponse(file_id=file_id)


@router.get("/models", response_model=ModelsDataReponse)
async def get_models() -> ModelsDataReponse:
    # Transform any known image model into ModelData object format and
    # store them as a list inside ModelsDataResponse
    return ModelsDataReponse(
        models=[ModelData.from_orm(model) for model in get_image_plugins().values()]
    )


@router.post("/process", response_model=TaskCreateResponse)
async def process_image(request: ImageProcessingRequest) -> TaskCreateResponse:
    created_task: TaskCreateResponse = await create_image_task(request)
    return created_task


@router.get("/download", response_class=FileResponse)
async def download_image_file(file: UUID) -> str:
    filepath = Path("./temp_data/image") / str(file)

    if not filepath.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not found"
        )

    return filepath.as_posix()


@router.get("/result", response_model=ImageProcessingResponse)
async def get_response(task_id: UUID) -> ImageProcessingResponse:
    response = await _get_job_status(task_id)
    if not response.ready:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail="The job is non-existent or not done",
        )

    job_results = _get_job_result(task_id)

    try:
        return ImageProcessingResponse.parse_obj(job_results.dict())
    except ValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="There is no such image processing task",
        ) from error
