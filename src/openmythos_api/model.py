from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import Settings


@dataclass(frozen=True)
class GenerationResult:
    text: str
    full_text: str
    input_tokens: int
    output_tokens: int
    model_variant: str
    device: str
    warning: str | None = None


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


class OpenMythosEngine:
    """Lazy OpenMythos loader and text generation wrapper."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = _resolve_device(settings.device)
        self.dtype = _resolve_dtype(settings.dtype, self.device)
        self.tokenizer: Any | None = None
        self.model: torch.nn.Module | None = None
        self.loaded = False
        self.warning: str | None = None

    def load(self) -> None:
        if self.loaded:
            return

        # Avoid importing OpenMythos at process start. This keeps /health fast and
        # makes tests possible without installing model dependencies.
        from open_mythos.main import MythosConfig, OpenMythos
        from open_mythos.tokenizer import MythosTokenizer

        self.tokenizer = MythosTokenizer(self.settings.tokenizer)
        vocab_size = self.settings.vocab_size or self.tokenizer.vocab_size
        cfg = self._build_config(MythosConfig, vocab_size)
        self.model = OpenMythos(cfg)

        if self.settings.weights_path:
            self._load_weights(self.settings.weights_path)
        else:
            self.warning = (
                "Default model is randomly initialized unless "
                "OPENMYTHOS_WEIGHTS_PATH is set."
            )

        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        self.loaded = True

    def _build_config(self, MythosConfig: Any, vocab_size: int) -> Any:
        variant = self.settings.model_variant.lower().strip()
        if variant != "tiny":
            return self._build_variant_config(variant)

        base: dict[str, Any] = {
            "vocab_size": vocab_size,
            "dim": self.settings.dim,
            "n_heads": self.settings.n_heads,
            "max_seq_len": self.settings.max_seq_len,
            "max_loop_iters": self.settings.max_loop_iters,
            "prelude_layers": self.settings.prelude_layers,
            "coda_layers": self.settings.coda_layers,
            "n_experts": self.settings.n_experts,
            "n_shared_experts": self.settings.n_shared_experts,
            "n_experts_per_tok": self.settings.n_experts_per_tok,
            "expert_dim": self.settings.expert_dim,
            "lora_rank": self.settings.lora_rank,
            "attn_type": self.settings.attn_type,
        }

        if self.settings.attn_type == "gqa":
            return MythosConfig(**base, n_kv_heads=self.settings.n_kv_heads)

        return MythosConfig(
            **base,
            n_kv_heads=self.settings.n_heads,
            kv_lora_rank=self.settings.kv_lora_rank,
            q_lora_rank=self.settings.q_lora_rank,
            qk_rope_head_dim=self.settings.qk_rope_head_dim,
            qk_nope_head_dim=self.settings.qk_nope_head_dim,
            v_head_dim=self.settings.v_head_dim,
        )

    def _build_variant_config(self, variant: str) -> Any:
        from open_mythos.variants import (
            mythos_1b,
            mythos_1t,
            mythos_3b,
            mythos_10b,
            mythos_50b,
            mythos_100b,
            mythos_500b,
        )

        variants = {
            "1b": mythos_1b,
            "3b": mythos_3b,
            "10b": mythos_10b,
            "50b": mythos_50b,
            "100b": mythos_100b,
            "500b": mythos_500b,
            "1t": mythos_1t,
        }
        if variant not in variants:
            allowed = ", ".join(["tiny", *variants.keys()])
            raise ValueError(f"Unknown OPENMYTHOS_MODEL_VARIANT={variant!r}. Use: {allowed}")
        return variants[variant]()

    def _load_weights(self, weights_path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model must be created before loading weights")

        path = Path(os.path.expanduser(weights_path))
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        self.model.load_state_dict(checkpoint, strict=False)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        n_loops: int,
        temperature: float,
        top_k: int | None,
        return_full_text: bool,
    ) -> GenerationResult:
        self.load()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("OpenMythos model failed to load")

        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            token_ids = [0]

        # Keep room for at least one generated token. For long prompts, use the tail.
        max_input_len = max(1, self.settings.max_seq_len - 1)
        token_ids = token_ids[-max_input_len:]
        input_len = len(token_ids)

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # OpenMythos generate supports these arguments in the public examples.
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "n_loops": n_loops,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_k is not None:
            kwargs["top_k"] = top_k

        output_ids = self.model.generate(input_ids, **kwargs)
        output_list = output_ids[0].detach().cpu().tolist()
        new_ids = output_list[input_len:]

        continuation = self.tokenizer.decode(new_ids)
        full_text = self.tokenizer.decode(output_list)

        return GenerationResult(
            text=full_text if return_full_text else continuation,
            full_text=full_text,
            input_tokens=input_len,
            output_tokens=len(new_ids),
            model_variant=self.settings.model_variant,
            device=str(self.device),
            warning=self.warning,
        )
