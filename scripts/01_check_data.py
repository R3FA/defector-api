#!/usr/bin/env python
"""Validate the MVTec AD dataset before any training runs.

Checks directory layout, image counts, defect type names, mask coverage and
resolutions for the three categories this project uses. Exits non-zero on any
mismatch, so a broken or partial dataset fails in seconds instead of halfway
through a training run.

Usage:
    python scripts/01_check_data.py
"""

import sys
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.table import Table

from defector.settings import settings

# Published figures from Bergmann et al. (CVPR 2019, Table 1; IJCV 2021, Table 8).
EXPECTED: dict[str, dict[str, object]] = {
    "screw": {
        "train_good": 320,
        "test_good": 41,
        "test_bad": 119,
        "defect_types": 5,
        "resolution": (1024, 1024),
    },
    "metal_nut": {
        "train_good": 220,
        "test_good": 22,
        "test_bad": 93,
        "defect_types": 4,
        "resolution": (700, 700),
    },
    "cable": {
        "train_good": 224,
        "test_good": 58,
        "test_bad": 92,
        "defect_types": 8,
        "resolution": (1024, 1024),
    },
}

IMAGE_SUFFIXES = {".png"}
console = Console(width=100)


def images_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def resolutions_of(paths: list[Path]) -> set[tuple[int, int]]:
    """Distinct (width, height). Pillow reads only the header, so this is cheap."""
    found = set()
    for path in paths:
        with Image.open(path) as image:
            found.add(image.size)
    return found


def inspect(category: str) -> tuple[dict[str, object], list[str]]:
    """Return measured facts plus a list of problems for one category."""
    expected = EXPECTED[category]
    root = settings.mvtec_dir / category
    problems: list[str] = []

    if not root.is_dir():
        return {"category": category}, [f"missing directory {category}/"]

    train_good = images_in(root / "train" / "good")
    test_good = images_in(root / "test" / "good")

    test_dir = root / "test"
    defect_dirs = (
        sorted(d for d in test_dir.iterdir() if d.is_dir() and d.name != "good")
        if test_dir.is_dir()
        else []
    )

    test_bad: list[Path] = []
    missing_masks: list[str] = []
    for defect_dir in defect_dirs:
        defect_images = images_in(defect_dir)
        test_bad.extend(defect_images)
        mask_dir = root / "ground_truth" / defect_dir.name
        for image_path in defect_images:
            if not (mask_dir / f"{image_path.stem}_mask.png").is_file():
                missing_masks.append(f"{defect_dir.name}/{image_path.name}")

    # --- counts -----------------------------------------------------------
    for label, actual, want in (
        ("train/good", len(train_good), expected["train_good"]),
        ("test/good", len(test_good), expected["test_good"]),
        ("test defective", len(test_bad), expected["test_bad"]),
        ("defect types", len(defect_dirs), expected["defect_types"]),
    ):
        if actual != want:
            problems.append(f"{label}: found {actual}, expected {want}")

    # --- masks ------------------------------------------------------------
    # Pixel AUROC and AUPRO (required by the thesis) are impossible without these.
    if missing_masks:
        shown = ", ".join(missing_masks[:3])
        more = f" (+{len(missing_masks) - 3} more)" if len(missing_masks) > 3 else ""
        problems.append(f"{len(missing_masks)} defective images without a mask: {shown}{more}")

    # --- resolutions ------------------------------------------------------
    resolutions = resolutions_of(train_good + test_good + test_bad)
    if resolutions and resolutions != {tuple(expected["resolution"])}:
        problems.append(
            f"resolutions {sorted(resolutions)}, expected {tuple(expected['resolution'])}"
        )

    facts = {
        "category": category,
        "train_good": len(train_good),
        "test_good": len(test_good),
        "test_bad": len(test_bad),
        "defect_types": [d.name for d in defect_dirs],
        "resolutions": sorted(resolutions),
        "masks_ok": not missing_masks,
    }
    return facts, problems


def main() -> int:
    console.print(f"[bold]MVTec AD[/bold]  {settings.mvtec_dir}")

    if not settings.mvtec_dir.is_dir() or not any(settings.mvtec_dir.iterdir()):
        console.print("[red]Dataset directory is missing or empty.[/red]")
        console.print("Download it with:\n")
        console.print(
            '  python -c "from anomalib.data import MVTecAD; '
            "MVTecAD(root='datasets/MVTecAD', category='screw').prepare_data()\"\n"
        )
        return 1

    table = Table(show_lines=False, pad_edge=False, padding=(0, 1))
    table.add_column("category", no_wrap=True)
    table.add_column("train", justify="right", no_wrap=True)
    table.add_column("test ok", justify="right", no_wrap=True)
    table.add_column("test nok", justify="right", no_wrap=True)
    table.add_column("types", justify="right", no_wrap=True)
    table.add_column("resolution", no_wrap=True)
    table.add_column("masks", no_wrap=True)

    all_problems: dict[str, list[str]] = {}
    for category in settings.categories:
        facts, problems = inspect(category)
        if problems:
            all_problems[category] = problems

        if "train_good" not in facts:
            table.add_row(category, "-", "-", "-", "-", "-", "[red]missing[/red]")
            continue

        resolution = ", ".join(f"{w}x{h}" for w, h in facts["resolutions"]) or "-"
        table.add_row(
            category,
            str(facts["train_good"]),
            str(facts["test_good"]),
            str(facts["test_bad"]),
            str(len(facts["defect_types"])),
            resolution,
            "[green]ok[/green]" if facts["masks_ok"] else "[red]missing[/red]",
        )

    console.print(table)

    for category in settings.categories:
        facts, _ = inspect(category)
        if facts.get("defect_types"):
            console.print(f"  [dim]{category}:[/dim] {', '.join(facts['defect_types'])}")

    if all_problems:
        console.print("\n[bold red]Problems found[/bold red]")
        for category, problems in all_problems.items():
            for problem in problems:
                console.print(f"  [red]{category}[/red]  {problem}")
        return 1

    total = sum(inspect(c)[0].get("train_good", 0) for c in settings.categories)
    console.print(
        f"\n[bold green]Dataset is valid.[/bold green] {total} training images across {len(settings.categories)} categories."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
