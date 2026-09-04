"""Project configuration.

All paths, categories, model hyperparameters and upload limits live here.
Any value can be overridden with a DEFECTOR_-prefixed environment variable
or a .env file, e.g. DEFECTOR_BACKBONE=wide_resnet50_2.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/defector/settings.py -> src/defector -> src -> repository root
ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEFECTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Categories -------------------------------------------------------
    # Order matters: the category classifier stores label indices in this order.
    categories: tuple[str, ...] = ("screw", "metal_nut", "cable")

    # --- Paths ------------------------------------------------------------
    datasets_dir: Path = ROOT / "datasets"
    mvtec_dir: Path = ROOT / "datasets" / "MVTecAD"
    results_dir: Path = ROOT / "results"
    artifacts_dir: Path = ROOT / "artifacts"
    uploads_dir: Path = ROOT / "storage" / "uploads"
    heatmaps_dir: Path = ROOT / "storage" / "heatmaps"
    database_path: Path = ROOT / "qc.db"

    # --- PatchCore --------------------------------------------------------
    # resnet18 for development (fast on CPU), wide_resnet50_2 for final numbers.
    backbone: Literal["resnet18", "wide_resnet50_2"] = "resnet18"
    layers: tuple[str, ...] = ("layer2", "layer3")
    # 1% scores the same average AUROC as 10% (Roth et al., Table S1) but is
    # roughly 10x faster to train and to run. Matters without a GPU.
    coreset_sampling_ratio: float = 0.01
    num_neighbors: int = 9
    # The MVTecAD datamodule has no image_size argument in anomalib 2.x;
    # these go through Patchcore.configure_pre_processor().
    image_size: tuple[int, int] = (256, 256)
    # Without the center crop the published numbers do not reproduce.
    center_crop_size: tuple[int, int] = (224, 224)

    # --- Training ---------------------------------------------------------
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 4
    seed: int = 42

    # --- Category classifier (linear probe on frozen features) ------------
    # Below this confidence the app asks the user to pick instead of guessing.
    category_confidence_threshold: float = 0.90
    classifier_path: Path = ROOT / "artifacts" / "category_classifier.joblib"

    # --- Upload -----------------------------------------------------------
    max_upload_bytes: int = 16 * 1024 * 1024
    allowed_image_formats: tuple[str, ...] = ("PNG", "JPEG", "BMP")
    # Decompression bomb guard; Pillow raises above twice this value.
    max_image_pixels: int = 40_000_000

    # --- CPU --------------------------------------------------------------
    # Leaves a couple of threads to the web layer.
    torch_num_threads: int = 12
    # The anyio default limiter is 40 threads, which oversubscribes 14 cores.
    inference_concurrency: int = 1

    def ensure_dirs(self) -> None:
        """Create the working directories; they are gitignored and absent after a clone."""
        for directory in (
            self.datasets_dir,
            self.results_dir,
            self.artifacts_dir,
            self.uploads_dir,
            self.heatmaps_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def category_results_dir(self, category: str) -> Path:
        return self.results_dir / category

    def checkpoint_path(self, category: str) -> Path:
        """Location of the Lightning checkpoint written by engine.fit()."""
        return (
            self.category_results_dir(category)
            / "Patchcore"
            / "MVTecAD"
            / category
            / "latest"
            / "weights"
            / "lightning"
            / "model.ckpt"
        )


settings = Settings()
