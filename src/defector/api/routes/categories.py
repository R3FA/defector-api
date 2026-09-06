"""Supported categories and their decision thresholds."""

from fastapi import APIRouter

from defector.api.contracts import CategoriesOut, CategoryInfo
from defector.ml.registry import registry
from defector.settings import settings

router = APIRouter(prefix="/api/categories", tags=["categories"])


@router.get("", response_model=CategoriesOut)
def list_categories() -> CategoriesOut:
    return CategoriesOut(
        categories=[
            CategoryInfo(name=name, threshold=threshold)
            for name, threshold in registry.thresholds().items()
        ],
        confidence_threshold=settings.category_confidence_threshold,
    )
