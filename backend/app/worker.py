from arq.connections import RedisSettings

from .config import settings
from . import storage, vectorstore
from .pipeline import ingest_document


async def on_startup(ctx: dict) -> None:
    await storage.ensure_bucket()
    await vectorstore.ensure_collections()


class WorkerSettings:
    functions = [ingest_document]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    on_startup = on_startup
    max_jobs = 2
    job_timeout = 900
