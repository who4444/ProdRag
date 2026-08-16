from fastapi import APIRouter
from fastapi.responses import Response

from ..core import storage

router = APIRouter()


@router.get("/files/{object_key:path}")
async def get_file(object_key: str):
    data = await storage.get_bytes(object_key)
    return Response(content=data, media_type="image/png")
