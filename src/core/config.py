import os
from pathlib import Path
from dotenv import load_dotenv


# Load .env from project root (two levels up from core/)
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / '.env')


class Config:
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION = os.getenv(
        "AZURE_OPENAI_API_VERSION", "2023-12-01-preview"
    )
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    # PostgreSQL Configuration (kept for compatibility)
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_NAME = os.getenv("DB_NAME", "rag_db")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    # Embedding Model
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    @classmethod
    def get_model_config(cls, model_key: str) -> dict:
        """Get Azure OpenAI config for a specific model.

        Looks up env vars with the pattern:
            AZURE_OPENAI_ENDPOINT_{MODEL_KEY}
            AZURE_OPENAI_API_KEY_{MODEL_KEY}
            API_VERSION_{MODEL_KEY}
            DEPLOYMENT_NAME_{MODEL_KEY}

        Falls back to the generic vars if model-specific
        ones are not found.
        """
        suffix = model_key.upper().replace("-", "_")
        return {
            "endpoint": (
                os.getenv(f"AZURE_OPENAI_ENDPOINT_{suffix}")
                or cls.AZURE_OPENAI_ENDPOINT
            ),
            "api_key": (
                os.getenv(f"AZURE_OPENAI_API_KEY_{suffix}")
                or cls.AZURE_OPENAI_API_KEY
            ),
            "api_version": (
                os.getenv(f"API_VERSION_{suffix}")
                or cls.AZURE_OPENAI_API_VERSION
            ),
            "deployment_name": (
                os.getenv(f"DEPLOYMENT_NAME_{suffix}")
                or cls.AZURE_OPENAI_DEPLOYMENT_NAME
            ),
        }

    @property
    def database_url(self):
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:"
            f"{self.DB_PORT}/{self.DB_NAME}"
        )
