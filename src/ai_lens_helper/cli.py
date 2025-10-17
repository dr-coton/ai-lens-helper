"""Command line interface for the ai-lens-helper toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from . import Lens
from .train.datamodule import DataValidator
from .train.engine import TrainingEngine, TrainingOptions
from .train.export import ExportOptions, ModelExporter
from .infer.runner import BatchInferenceOptions, InferenceOptions
from .infer.index import IndexBuilder, IndexOptions
from .data.crawler import (
    load_crawl_specs_from_json,
    save_crawl_report,
)
from .data.selenium_crawler import SeleniumImageCrawler
from .data.naver_crawler import NaverImageCrawler
from .data.fast_naver_crawler import FastNaverImageCrawler

app = typer.Typer(
    help="Hybrid classification + retrieval helper for museum exhibit recognition.",
    no_args_is_help=True,
)
console = Console()


@app.command("validate-data")
def validate_data(
    data_root: Path = typer.Argument(..., exists=True, file_okay=False, resolve_path=True),
    min_images: int = typer.Option(10, help="Minimum number of images required per exhibit."),
) -> None:
    """Validate the training dataset layout and provide a short report."""

    validator = DataValidator(data_root=data_root, min_images=min_images)
    report = validator.validate()
    table = Table(title="Dataset summary", box=box.SIMPLE_HEAVY)
    table.add_column("Place", justify="left")
    table.add_column("Items", justify="right")
    table.add_column(">= min", justify="right")
    for summary in report.place_summaries:
        table.add_row(summary.place, str(summary.item_count), str(summary.items_meeting_requirement))
    console.print(table)
    if report.issues:
        console.print("[bold yellow]Warnings detected during validation:[/bold yellow]")
        for issue in report.issues:
            console.print(f" - {issue}")
    else:
        console.print("[bold green]Dataset layout looks good![/bold green]")


@app.command()
def train(
    config: Optional[Path] = typer.Option(None, "--config", exists=True, resolve_path=True),
    data_root: Path = typer.Option(..., exists=True, file_okay=False, resolve_path=True),
    output_dir: Path = typer.Option(Path("./runs"), resolve_path=True),
) -> None:
    """Run the training pipeline (placeholder implementation)."""

    options = TrainingOptions(config_path=config, data_root=data_root, output_dir=output_dir)
    engine = TrainingEngine(options=options)
    message = engine.train()
    console.print(f"[bold cyan]{message}[/bold cyan]")


@app.command()
def infer(
    model: Path = typer.Option(..., exists=True, resolve_path=True),
    place: str = typer.Option(..., help="Museum place identifier"),
    image: Path = typer.Option(..., exists=True, resolve_path=True, help="Image to classify"),
    reject_threshold: Optional[float] = typer.Option(None, help="Override reject threshold"),
    topk: int = typer.Option(3, min=1, max=10),
) -> None:
    """Perform single image inference using the SDK facade."""

    lens = Lens(model_path=model)
    options = InferenceOptions(place=place, image_path=image, reject_threshold=reject_threshold, topk=topk)
    result = lens.infer(
        place=options.place,
        image_path=options.image_path,
        reject_threshold=options.reject_threshold,
        topk=options.topk,
    )
    console.print_json(data=result.as_dict())


@app.command("infer-batch")
def infer_batch(
    model: Path = typer.Option(..., exists=True, resolve_path=True),
    place: str = typer.Option(...),
    input_dir: Path = typer.Option(..., exists=True, resolve_path=True, file_okay=False),
    output: Path = typer.Option(Path("./batch_results.jsonl"), resolve_path=True),
    reject_threshold: Optional[float] = typer.Option(None),
    topk: int = typer.Option(3, min=1, max=10),
) -> None:
    """Placeholder batch inference command."""

    options = BatchInferenceOptions(
        model_path=model,
        place=place,
        input_dir=input_dir,
        output_path=output,
        reject_threshold=reject_threshold,
        topk=topk,
    )
    console.print(options.summary_message())


@app.command("build-index")
def build_index(
    model: Path = typer.Option(..., exists=True, resolve_path=True),
    data_root: Path = typer.Option(..., exists=True, resolve_path=True, file_okay=False),
    place: str = typer.Option(...),
    save: Path = typer.Option(..., resolve_path=True),
) -> None:
    """Create a retrieval index (structure only for now)."""

    options = IndexOptions(model_path=model, data_root=data_root, place=place, save_path=save)
    builder = IndexBuilder(options=options)
    summary = builder.build()
    console.print(
        "[bold green]Index ready[/bold green]: "
        f"place='{summary.place}', items={summary.item_count}, images={summary.image_count}, "
        f"path='{summary.save_path}'"
    )


@app.command("build-clip-index")
def build_clip_index(
    data_root: Path = typer.Option(..., exists=True, resolve_path=True, file_okay=False),
    place: str = typer.Option(..., help="Place identifier to index"),
    save: Path = typer.Option(..., resolve_path=True, help="Output path for index (without extension)"),
    device: str = typer.Option("cpu", help="Device to use (cpu or cuda)"),
    yolo_model: str = typer.Option("yolov8n.pt", help="YOLO model variant (yolov8n/s/m/l/x)"),
    clip_model: str = typer.Option("ViT-B-16", help="CLIP model architecture"),
) -> None:
    """Build YOLO+CLIP based FAISS index from training data.

    This command:
    1. Loads all images from data_root/{place}/{item}/*.jpg
    2. Uses YOLO to detect and crop exhibits
    3. Extracts CLIP embeddings for each image
    4. Builds a FAISS index for fast similarity search

    The output will be two files:
    - {save}.faiss: FAISS index for nearest neighbor search
    - {save}.json: Metadata with item names and embeddings
    """
    from .models.yolo_clip import YOLOCLIPPipeline
    from .infer.clip_index import CLIPIndexBuilder

    console.print(f"[bold cyan]Building YOLO+CLIP index for place '{place}'[/bold cyan]")
    console.print(f"Data root: {data_root}")
    console.print(f"Device: {device}")
    console.print(f"YOLO model: {yolo_model}")
    console.print(f"CLIP model: {clip_model}")
    console.print()

    # Initialize pipeline
    console.print("Initializing YOLO+CLIP pipeline...")
    pipeline = YOLOCLIPPipeline(
        yolo_model=yolo_model,
        clip_model=clip_model,
        clip_pretrained="openai",
        device=device
    )

    # Build index
    console.print(f"Processing images from {data_root}/{place}...")
    builder = CLIPIndexBuilder(pipeline=pipeline)
    index_items = builder.build_from_directory(data_root=data_root, place=place)

    if not index_items:
        console.print("[bold red]No items found to index![/bold red]")
        return

    # Save index
    builder.save_index(
        index_items=index_items,
        output_path=save,
        place=place,
        metadata={"reject_threshold": 0.7}
    )

    total_images = sum(item.image_count for item in index_items.values())
    console.print()
    console.print(f"[bold green]✓ Index built successfully[/bold green]")
    console.print(f"  Place: {place}")
    console.print(f"  Items: {len(index_items)}")
    console.print(f"  Total images: {total_images}")


@app.command()
def export(
    ckpt: Path = typer.Option(..., exists=True, resolve_path=True),
    onnx: Optional[Path] = typer.Option(None, resolve_path=True),
    opset: int = typer.Option(17, min=10, max=20),
) -> None:
    """Export a trained checkpoint to deployment formats (stub)."""

    options = ExportOptions(ckpt_path=ckpt, onnx_path=onnx, opset=opset)
    exporter = ModelExporter(options=options)
    console.print(exporter.export())


@app.command("collect-data")
def collect_data(
    spec_json: Path = typer.Argument(..., exists=True, resolve_path=True),
    output_root: Path = typer.Option(
        Path("./data"), resolve_path=True, help="Root directory for downloaded images"
    ),
    num_images: int = typer.Option(20, min=1, max=100, help="Number of images per item"),
    headless: bool = typer.Option(True, help="Run browser in headless mode"),
    scroll_pause: float = typer.Option(1.0, min=0.5, max=5.0, help="Pause after scrolling (seconds)"),
    download_delay: float = typer.Option(
        0.3, min=0.1, max=2.0, help="Delay between downloads (seconds)"
    ),
    engine: str = typer.Option(
        "naver", help="Search engine to use: 'naver' or 'google'"
    ),
    report: Optional[Path] = typer.Option(None, resolve_path=True, help="Save crawl report to JSON"),
) -> None:
    """Collect test images by crawling image search engines using Selenium.

    The spec_json should contain place-item mappings in one of these formats:

    Simple format:
    {
        "place_A": ["item1", "item2"],
        "place_B": ["item3", "item4"]
    }

    Detailed format (with custom num_images per item):
    {
        "place_A": [
            {"item": "item1", "num_images": 20},
            {"item": "item2", "num_images": 15}
        ]
    }

    This command uses Selenium WebDriver to:
    - Handle dynamic JavaScript content
    - Scroll and load more images
    - Extract high-resolution image URLs
    - Download images with proper error handling

    Supported engines:
    - naver: Naver Images (recommended for Korean content)
    - google: Google Images
    """
    # Load crawl specifications
    specs = load_crawl_specs_from_json(spec_json)

    # Override num_images if provided via CLI (only for simple format)
    for spec in specs:
        if spec.num_images == 20:  # Default value, can be overridden
            spec.num_images = num_images

    # Show summary
    console.print(f"[bold cyan]Starting {engine.title()}-based data collection[/bold cyan]")
    console.print(f"Total items: {len(specs)}")
    console.print(f"Output root: {output_root}")
    console.print(f"Images per item: {num_images} (default)")
    console.print(f"Headless mode: {headless}")
    console.print()

    # Create appropriate crawler
    if engine.lower() == "naver":
        # Use Selenium-based crawler with proxy URL extraction
        crawler = NaverImageCrawler(
            output_root=output_root,
            headless=headless,
            scroll_pause=scroll_pause,
            download_delay=download_delay,
        )
    elif engine.lower() == "google":
        crawler = SeleniumImageCrawler(
            output_root=output_root,
            headless=headless,
            scroll_pause=scroll_pause,
            download_delay=download_delay,
        )
    else:
        console.print(f"[bold red]Unknown engine: {engine}[/bold red]")
        console.print("Supported engines: naver, google")
        return

    results = crawler.crawl_batch(specs)

    # Print summary table
    table = Table(title="Collection Summary", box=box.SIMPLE_HEAVY)
    table.add_column("Place", justify="left")
    table.add_column("Item", justify="left")
    table.add_column("Downloaded", justify="right")
    table.add_column("Failed", justify="right")

    for result in results:
        table.add_row(
            result.place,
            result.item,
            str(result.downloaded),
            str(result.failed),
        )

    console.print(table)

    # Print overall statistics
    total_downloaded = sum(r.downloaded for r in results)
    total_failed = sum(r.failed for r in results)
    console.print()
    console.print(f"[bold green]Total downloaded: {total_downloaded}[/bold green]")
    if total_failed > 0:
        console.print(f"[bold yellow]Total failed: {total_failed}[/bold yellow]")

    # Save report if requested
    if report:
        save_crawl_report(results, report)
        console.print(f"[bold cyan]Report saved to: {report}[/bold cyan]")


if __name__ == "__main__":  # pragma: no cover - script entry point
    app()
