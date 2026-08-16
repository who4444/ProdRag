from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRODRAG_", env_file=".env", extra="ignore"
    )

    api_token: str = "change-me"

    # Answering LLM. DeepSeek is OpenAI-compatible at api.deepseek.com and is
    # chat-only (no vision, no embeddings). Leave chat_base_url empty for OpenAI.
    chat_api_key: str = ""
    chat_base_url: str = "https://api.deepseek.com"
    chat_model: str = "deepseek-v4-flash"
    chat_supports_images: bool = False
    max_images_in_context: int = 3

    # Text embeddings: default to self-hosted bge-m3 via Modal (1024 dims).
    # openai_api_key is used only for the optional text-embedding-3-small
    # fallback when embed_service_url is unset — then set embedding_dim=1536.
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1024
    embed_service_url: str = ""
    embed_service_token: str = ""

    vision_embedding_model: str = "ViT-B-32"
    vision_embedding_pretrained: str = "laion2b_s34b_b79k"
    vision_dim: int = 512
    clip_service_url: str = ""
    clip_service_token: str = ""

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection_text: str = "text_chunks"
    collection_image: str = "image_chunks"

    redis_url: str = "redis://localhost:6379"

    # Object storage via S3. Primary: Supabase Storage (S3-compatible at
    # {supabase_url}/storage/v1/s3, keys from Project Settings > Storage).
    # PRODRAG_S3_ENDPOINT overrides the endpoint for any other S3-compatible store.
    supabase_url: str = ""
    s3_endpoint: str = ""
    s3_region: str = "us-east-1"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    storage_bucket: str = "prodrag-assets"

    chunk_size: int = 800
    chunk_overlap: int = 100
    page_dpi: int = 150
    min_figure_area: float = 0.04

    # Research agent + memory
    agent_model: str = "deepseek-v4-flash"
    agent_max_steps: int = 12
    memory_collection: str = "episodes"
    memory_top_k: int = 5
    memory_ttl_s: int = 86400


settings = Settings()
