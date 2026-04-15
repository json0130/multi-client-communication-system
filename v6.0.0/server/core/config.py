# Reads its configuration from. No other file will call os.getenv directly.

"""
core/config.py
==============
Single source of truth for all configuration.
Every other module imports from here — nothing else calls os.getenv directly.

Usage:
    from core.config import cfg
    print(cfg.supabase_url)
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class DatabaseConfig:
    url: str
    key: str


@dataclass(frozen=True)
class LLMConfig:
    # Ollama (local)
    ollama_host: str
    ollama_port: int
    ollama_model: str
    # OpenAI (cloud fallback, optional)
    openai_api_key: Optional[str]
    openai_model: str


@dataclass(frozen=True)
class SpeechConfig:
    whisper_model_size: str      # tiny | base | small | medium | large
    whisper_device: str          # auto | cpu | cuda
    whisper_compute_type: str    # int8 | float16 | float32
    max_audio_length_sec: int
    sample_rate: int


@dataclass(frozen=True)
class EmotionConfig:
    model_path: str
    processing_interval_sec: float
    frame_skip_ratio: int
    confidence_threshold: float
    emotion_change_threshold: float
    emotion_update_threshold: float
    emotion_window_size: int


@dataclass(frozen=True)
class RagConfig:
    index_dir: str
    embed_model: str             # Ollama embedding model name
    ollama_host: str
    ollama_port: int


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    cors_origins: str


@dataclass(frozen=True)
class AppConfig:
    db: DatabaseConfig
    llm: LLMConfig
    speech: SpeechConfig
    emotion: EmotionConfig
    rag: RagConfig
    server: ServerConfig


def _require(key: str) -> str:
    """Read a required env var; raise clearly if missing."""
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Add it to your .env file."
        )
    return val


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def load_config() -> AppConfig:
    """
    Build and return the full AppConfig.
    Called once at startup; result stored in module-level `cfg`.
    """
    return AppConfig(
        db=DatabaseConfig(
            url=_require("SUPABASE_URL"),
            key=_require("SUPABASE_KEY"),
        ),
        llm=LLMConfig(
            ollama_host=_optional("OLLAMA_HOST", "127.0.0.1"),
            ollama_port=_int("OLLAMA_PORT", 11434),
            ollama_model=_optional("OLLAMA_MODEL", "qwen2.5:7b"),
            openai_api_key=_optional("OPENAI_API_KEY") or None,
            openai_model=_optional("OPENAI_MODEL", "gpt-4o-mini"),
        ),
        speech=SpeechConfig(
            whisper_model_size=_optional("WHISPER_MODEL_SIZE", "base"),
            whisper_device=_optional("WHISPER_DEVICE", "auto"),
            whisper_compute_type=_optional("WHISPER_COMPUTE_TYPE", "int8"),
            max_audio_length_sec=_int("WHISPER_MAX_AUDIO_SEC", 30),
            sample_rate=_int("WHISPER_SAMPLE_RATE", 16000),
        ),
        emotion=EmotionConfig(
            model_path=_optional(
                "EMOTION_MODEL_PATH",
                "./models/efficientnet_HQRAF_improved_withCon.pth"
            ),
            processing_interval_sec=_float("EMOTION_INTERVAL", 0.2),
            frame_skip_ratio=_int("EMOTION_FRAME_SKIP", 3),
            confidence_threshold=_float("EMOTION_CONFIDENCE_THRESHOLD", 25.0),
            emotion_change_threshold=_float("EMOTION_CHANGE_THRESHOLD", 20.0),
            emotion_update_threshold=_float("EMOTION_UPDATE_THRESHOLD", 0.1),
            emotion_window_size=_int("EMOTION_WINDOW_SIZE", 3),
        ),
        rag=RagConfig(
            index_dir=_optional("RAG_INDEX_DIR", "./rag_indexes"),
            embed_model=_optional("RAG_EMBED_MODEL", "nomic-embed-text"),
            ollama_host=_optional("OLLAMA_HOST", "127.0.0.1"),
            ollama_port=_int("OLLAMA_PORT", 11434),
        ),
        server=ServerConfig(
            host=_optional("SERVER_HOST", "0.0.0.0"),
            port=_int("SERVER_PORT", 5000),
            cors_origins=_optional("CORS_ORIGINS", "*"),
        ),
    )


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this anywhere: `from core.config import cfg`
try:
    cfg: AppConfig = load_config()
except EnvironmentError as _e:
    # Let the app boot in partial mode during testing; individual modules
    # will fail loudly when they actually try to use missing credentials.
    import warnings
    warnings.warn(str(_e))
    cfg = None  # type: ignore