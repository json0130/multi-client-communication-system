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
from dataclasses import dataclass
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
    language: str                # en | ... | auto (detect per utterance)


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
    api_token: str   # required on every non-GET request when set; "" = open
    inbound_port: int = 8790   # robots that dial in to the server (client_config connect_to_server); 0 = off


@dataclass(frozen=True)
class RolesConfig:
    """The Roles page (gateway/roles_gateway.py): assign role + persona to a robot, persona chat, graph."""
    port: int                    # 0 = off
    data_dir: str                # its graph / persona / registry files
    ai_server: str               # "HOST[:PORT]" persona-chat backend; "" = Ollama above
    sync_db: bool                # an assigned role also becomes robots.robot_role (what the server's own turns use)


@dataclass(frozen=True)
class MethodBConfig:
    """Method B (conversation/method_b.py): a gesture tag per Navel sentence, planned in the server."""
    ollama_model: str            # tagging model on the Ollama above (default: the LLM's); "off" with no ai_server = off
    ai_server: str               # "HOST[:PORT]" instead of Ollama
    timeout_sec: float           # the first sentence waits this long for its tag; later ones, until the earlier ones are spoken


@dataclass(frozen=True)
class AppConfig:
    db: DatabaseConfig
    llm: LLMConfig
    speech: SpeechConfig
    emotion: EmotionConfig
    rag: RagConfig
    server: ServerConfig
    roles: RolesConfig
    method_b: MethodBConfig


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
            language=_optional("WHISPER_LANGUAGE", "en"),
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
            api_token=_optional("API_TOKEN", ""),
            inbound_port=_int("ROBOT_INBOUND_PORT", 8790),
        ),
        roles=RolesConfig(
            port=_int("ROLES_PORT", 8780),
            data_dir=_optional("ROLES_DATA_DIR", "./roles_data"),
            ai_server=_optional("ROLES_AI_SERVER", ""),
            sync_db=_optional("ROLES_SYNC_DB", "0") == "1",
        ),
        method_b=MethodBConfig(
            ollama_model=_optional("METHOD_B_OLLAMA_MODEL", _optional("OLLAMA_MODEL", "qwen2.5:7b")),
            ai_server=_optional("METHOD_B_AI_SERVER", ""),
            timeout_sec=_float("METHOD_B_TIMEOUT_SEC", 1.0),
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