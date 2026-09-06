from fastapi import APIRouter, Depends, Header, HTTPException

from ..config import settings

router = APIRouter()


async def require_token(authorization: str = Header(default="")):
    if authorization != f"Bearer {settings.api_token}":
        raise HTTPException(status_code=401, detail="invalid token")


from . import code, conversations, documents, files, memory, query, rd, research, tools  # noqa: E402

router.include_router(documents.router, dependencies=[Depends(require_token)])
router.include_router(files.router, dependencies=[Depends(require_token)])
router.include_router(query.router, dependencies=[Depends(require_token)])
router.include_router(research.router, dependencies=[Depends(require_token)])
router.include_router(rd.router, dependencies=[Depends(require_token)])
router.include_router(tools.router, dependencies=[Depends(require_token)])
router.include_router(code.router, dependencies=[Depends(require_token)])
router.include_router(conversations.router, dependencies=[Depends(require_token)])
router.include_router(memory.router, dependencies=[Depends(require_token)])
