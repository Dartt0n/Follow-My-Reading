from functools import lru_cache
from pathlib import Path

from pydantic import BaseConfig, BaseSettings, Field


class Config(BaseSettings):
    class Config(BaseConfig):
        env_file = ".env"

    class Redis(BaseSettings):
        host: str = "localhost"
        port: int = 6379
        url: str = f"redis:{host}:{port}"

    class Storage(BaseSettings):
        files_dir: Path = Path("temp_data")
        image_dir: Path = files_dir / "image"
        audio_dir: Path = files_dir / "audio"

    class Token(BaseSettings):
        secket_key: str = Field(..., env="SECRET_KEY")
        jwt_algorithm: str = "HS256"
        access_expire_minutes: int = 30

    redis: Redis
    storage: Storage
    token: Token


@lru_cache
def get_config() -> Config:
    if Path(".config.json").exists():
        return Config.parse_file(".config.json")  # read settings from .config.json file
    return Config.parse_file("example.config.json")
