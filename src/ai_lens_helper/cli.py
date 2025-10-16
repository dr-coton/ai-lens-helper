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


if __name__ == "__main__":  # pragma: no cover - script entry point
    app()
