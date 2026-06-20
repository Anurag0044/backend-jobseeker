from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    GEMINI_API_KEY: str
    FIREBASE_SERVICE_ACCOUNT_JSON: str

    @field_validator("GEMINI_API_KEY")
    @classmethod
    def gemini_key_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "GEMINI_API_KEY is required but is empty. "
                "Set it in your .env file."
            )
        return v

    @field_validator("FIREBASE_SERVICE_ACCOUNT_JSON")
    @classmethod
    def firebase_json_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "FIREBASE_SERVICE_ACCOUNT_JSON is required but is empty. "
                "Paste your minified Firebase service account JSON in .env."
            )
        return v


settings = Settings()
