from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .model import GenerationResult, OpenMythosEngine


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=20000)
    max_new_tokens: int = Field(default=64, ge=1, le=4096)
    n_loops: int = Field(default=4, ge=1, le=128)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_k: int | None = Field(default=40, ge=1, le=1000)
    return_full_text: bool = False


class GenerateResponse(BaseModel):
    text: str
    full_text: str
    input_tokens: int
    output_tokens: int
    model_variant: str
    device: str
    warning: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_variant: str
    device: str
    auth_enabled: bool


app = FastAPI(
    title="OpenMythos GPT / Claude API Bridge",
    version="0.1.0",
    description="Call OpenMythos from Custom GPT Actions, Claude tool use, or any REST client.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@lru_cache(maxsize=1)
def get_engine() -> OpenMythosEngine:
    return OpenMythosEngine(get_settings())


async def require_auth(
    request: Request, settings: Settings = Depends(get_settings)
) -> None:
    if not settings.api_key:
        return

    expected = f"Bearer {settings.api_key}"
    received = request.headers.get("Authorization")
    if received != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token",
        )


@app.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_variant=settings.model_variant,
        device=settings.device,
        auth_enabled=bool(settings.api_key),
    )


@app.post(
    "/generate",
    response_model=GenerateResponse,
    dependencies=[Depends(require_auth)],
)
def generate(
    payload: GenerateRequest,
    settings: Settings = Depends(get_settings),
    engine: OpenMythosEngine = Depends(get_engine),
) -> GenerateResponse:
    if payload.max_new_tokens > settings.max_new_tokens_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "max_new_tokens exceeds server limit "
                f"{settings.max_new_tokens_limit}"
            ),
        )

    try:
        result: GenerationResult = engine.generate(
            prompt=payload.prompt,
            max_new_tokens=payload.max_new_tokens,
            n_loops=payload.n_loops,
            temperature=payload.temperature,
            top_k=payload.top_k,
            return_full_text=payload.return_full_text,
        )
    except Exception as exc:  # noqa: BLE001 - converted to API error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OpenMythos generation failed: {exc}",
        ) from exc

    return GenerateResponse(**result.__dict__)
