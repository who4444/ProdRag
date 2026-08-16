from fastapi import APIRouter, Depends, Header, HTTPException

from ..config import settings

router = APIRouter()


async def require_token(authorization: str = Header(default="")):
    if authorization != f"Bearer {settings.api_token}":
        raise HTTPException(status_code=401, detail="invalid token")


from . import documents, files, query, research  # noqa: E402

router.include_router(documents.router, dependencies=[Depends(require_token)])
router.include_router(files.router, dependencies=[Depends(require_token)])
router.include_router(query.router, dependencies=[Depends(require_token)])
router.include_router(research.router, dependencies=[Depends(require_token)])
