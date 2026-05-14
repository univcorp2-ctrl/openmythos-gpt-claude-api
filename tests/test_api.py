from dataclasses import dataclass

from fastapi.testclient import TestClient

from openmythos_api.config import Settings, get_settings
from openmythos_api.main import app, get_engine


@dataclass(frozen=True)
class FakeResult:
    text: str
    full_text: str
    input_tokens: int
    output_tokens: int
    model_variant: str
    device: str
    warning: str | None = None


class FakeEngine:
    def generate(self, **kwargs):
        return FakeResult(
            text="fake continuation",
            full_text=kwargs["prompt"] + " fake continuation",
            input_tokens=2,
            output_tokens=2,
            model_variant="tiny",
            device="cpu",
            warning="test warning",
        )


def override_settings_no_auth() -> Settings:
    return Settings(api_key=None, max_new_tokens_limit=128)


def override_settings_auth() -> Settings:
    return Settings(api_key="secret", max_new_tokens_limit=128)


def setup_function():
    app.dependency_overrides.clear()
    app.dependency_overrides[get_engine] = lambda: FakeEngine()


def teardown_function():
    app.dependency_overrides.clear()


def test_health_without_auth():
    app.dependency_overrides[get_settings] = override_settings_no_auth
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["auth_enabled"] is False


def test_generate_without_auth():
    app.dependency_overrides[get_settings] = override_settings_no_auth
    client = TestClient(app)

    response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "fake continuation"
    assert body["warning"] == "test warning"


def test_generate_rejects_missing_token_when_auth_enabled():
    app.dependency_overrides[get_settings] = override_settings_auth
    client = TestClient(app)

    response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 401


def test_generate_accepts_bearer_token():
    app.dependency_overrides[get_settings] = override_settings_auth
    client = TestClient(app)

    response = client.post(
        "/generate",
        json={"prompt": "hello"},
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200


def test_generate_respects_server_token_limit():
    app.dependency_overrides[get_settings] = override_settings_no_auth
    client = TestClient(app)

    response = client.post(
        "/generate",
        json={"prompt": "hello", "max_new_tokens": 129},
    )

    assert response.status_code == 400
