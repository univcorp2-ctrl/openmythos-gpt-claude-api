from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the OpenMythos API bridge."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="OPENMYTHOS_",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: str | None = Field(default=None, description="Optional Bearer token")
    device: str = Field(default="auto", description="auto, cpu, cuda, or mps")
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    model_variant: str = Field(default="tiny", description="tiny, 1b, 3b, ...")
    attn_type: Literal["gqa", "mla"] = "gqa"
    tokenizer: str = Field(default="openai/gpt-oss-20b")
    weights_path: str | None = None

    # Tiny config defaults. These only apply when model_variant=tiny.
    vocab_size: int | None = None
    dim: int = 256
    n_heads: int = 8
    n_kv_heads: int = 2
    max_seq_len: int = 512
    max_loop_iters: int = 4
    prelude_layers: int = 1
    coda_layers: int = 1
    n_experts: int = 8
    n_shared_experts: int = 1
    n_experts_per_tok: int = 2
    expert_dim: int = 64
    lora_rank: int = 4

    # MLA-only tiny config fields.
    kv_lora_rank: int = 32
    q_lora_rank: int = 64
    qk_rope_head_dim: int = 16
    qk_nope_head_dim: int = 16
    v_head_dim: int = 16

    max_new_tokens_limit: int = 256


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
