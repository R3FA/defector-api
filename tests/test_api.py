"""End-to-end API behaviour, including regressions for two fixed bugs."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from defector.api.contracts import AmbiguousCategoryResponse
from defector.settings import settings


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Starts the app for real, so the lifespan loads the trained models."""
    from defector.api.app import app

    with TestClient(app) as client:
        yield client


def _noise_png() -> bytes:
    """An image that is none of the three categories, so routing must refuse it."""
    rng = np.random.default_rng(seed=7)
    buffer = io.BytesIO()
    Image.fromarray((rng.random((256, 256, 3)) * 255).astype("uint8")).save(buffer, format="PNG")
    return buffer.getvalue()


def _sample(category: str, subdirectory: str) -> bytes:
    path = next((settings.mvtec_dir / category / "test" / subdirectory).glob("*.png"))
    return path.read_bytes()


def _stored() -> tuple[int, int]:
    return (
        len(list(settings.uploads_dir.glob("*"))),
        len(list(settings.heatmaps_dir.glob("*"))),
    )


def test_health_reports_the_loaded_categories(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert set(body["categories"]) == set(settings.categories)


def test_analyse_routes_and_scores_a_known_defect(client: TestClient) -> None:
    response = client.post(
        "/api/analyze", files={"file": ("x.png", _sample("metal_nut", "bent"), "image/png")}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["category"] == "metal_nut"
    assert body["is_defective"] is True
    assert body["category_auto"] is True
    assert body["created_at"].endswith("+00:00")  # UTC offset, not a naive stamp


def test_explicit_category_skips_routing(client: TestClient) -> None:
    body = client.post(
        "/api/analyze",
        files={"file": ("x.png", _sample("cable", "good"), "image/png")},
        data={"category": "cable"},
    ).json()
    assert body["category_auto"] is False
    assert body["category_confidence"] is None


def test_unroutable_image_asks_the_caller_to_choose(client: TestClient) -> None:
    response = client.post("/api/analyze", files={"file": ("n.png", _noise_png(), "image/png")})
    assert response.status_code == 409
    # The body must match what /docs advertises for 409.
    AmbiguousCategoryResponse.model_validate(response.json())
    detail = response.json()["detail"]
    assert set(detail["probabilities"]) == set(settings.categories)
    assert detail["confidence"] < settings.category_confidence_threshold


def test_a_refused_upload_leaves_no_file_behind(client: TestClient) -> None:
    """Regression: the original used to be written before routing was attempted,
    so every 409 orphaned a file that no row ever referenced."""
    before = _stored()
    assert (
        client.post(
            "/api/analyze", files={"file": ("n.png", _noise_png(), "image/png")}
        ).status_code
        == 409
    )
    assert _stored() == before


def test_rejects_a_non_image(client: TestClient) -> None:
    response = client.post("/api/analyze", files={"file": ("x.png", b"not an image", "image/png")})
    assert response.status_code == 400
    assert "not a readable image" in response.json()["detail"]


def test_rejects_an_unknown_category(client: TestClient) -> None:
    response = client.post(
        "/api/analyze",
        files={"file": ("x.png", _sample("screw", "good"), "image/png")},
        data={"category": "tocak"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/api/analyses/0", 422),
        ("/api/analyses/-1", 422),
        ("/api/analyses/abc", 422),
        # Regression: an id past SQLite's 64-bit INTEGER used to reach the driver
        # and surface as a 500 instead of a validation error.
        ("/api/analyses/99999999999999999999", 422),
        ("/api/analyses/999999999", 404),
    ],
)
def test_identifier_bounds(client: TestClient, path: str, expected: int) -> None:
    assert client.get(path).status_code == expected


@pytest.mark.parametrize(
    "query",
    ["?limit=0", "?limit=201", "?offset=-1", "?defective=nesto"],
)
def test_invalid_query_parameters_are_rejected(client: TestClient, query: str) -> None:
    assert client.get(f"/api/analyses{query}").status_code == 422


def test_stats_is_not_parsed_as_an_identifier(client: TestClient) -> None:
    assert client.get("/api/analyses/stats").status_code == 200


def test_stored_images_are_served_for_a_recorded_analysis(client: TestClient) -> None:
    created = client.post(
        "/api/analyze", files={"file": ("x.png", _sample("screw", "good"), "image/png")}
    ).json()

    for url in (created["original_url"], created["heatmap_url"]):
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.headers.get("etag")


def test_categories_expose_the_decision_thresholds(client: TestClient) -> None:
    body = client.get("/api/categories").json()
    assert {entry["name"] for entry in body["categories"]} == set(settings.categories)
    assert body["confidence_threshold"] == settings.category_confidence_threshold
