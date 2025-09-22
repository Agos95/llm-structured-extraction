import streamlit as st
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        extra="ignore",
    )

    AWS_REGION: str | None = None
    AWS_PROFILE: str | None = None
    LLM: str


@st.cache_resource
def _settings():
    return Settings()


settings = _settings()
