from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import typer

from .cli_helpers import (
    build_log_writer,
    default_annotate_output_path,
    run_face_stage,
)
from .face_embedder import FaceEmbedder
from .face_pipeline import FacePipeline
from .identity_config import IdentityConfig
from .identity_engine import IdentityEngine
from .identity_store import IdentityStore

app = typer.Typer(add_completion=False, help="Face-based identity pipeline.")


def _build_identity(store_path: Path) -> IdentityEngine:
    store = IdentityStore(store_path)
    config = IdentityConfig()
    return IdentityEngine(store=store, config=config)


def _review_data_paths(data_dir: Path) -> tuple[Path, Path, Path]:
    nested_identity = data_dir / "profiles" / "face_templates.json"
    if nested_identity.exists() and not (data_dir / "face_templates.json").exists():
        return (
            nested_identity,
            data_dir / "profiles" / "beverage_events.json",
            data_dir / "events" / "decisions.jsonl",
        )
    return (
        data_dir / "face_templates.json",
        data_dir / "beverage_events.json",
        data_dir / "decisions.jsonl",
    )


@app.command()
def identify(
    source: str = typer.Argument(..., help="Path to video file or camera index."),
    store_path: Path = typer.Option(
        Path("profiles/face_templates.json"), "--store", help="Template store path."
    ),
    limit_frames: Optional[int] = typer.Option(
        None, "--limit-frames", help="Stop after N frames for quick checks."
    ),
    verbose: bool = typer.Option(
        True, "--verbose/--quiet", help="Print per-frame processing updates."
    ),
    update_templates: bool = typer.Option(
        False,
        "--update-templates/--no-update-templates",
        help="Update templates on high-confidence matches.",
    ),
    annotate_output: Optional[Path] = typer.Option(
        None,
        "--annotate-output",
        help="Write annotated output video with bounding boxes and identified user IDs.",
    ),
    auto_enroll_unknown: bool = typer.Option(
        False,
        "--auto-enroll-unknown/--no-auto-enroll-unknown",
        help="Automatically enroll unknown identities with generated user IDs.",
    ),
    tee_logs: bool = typer.Option(
        False,
        "--tee-logs/--no-tee-logs",
        help="Also mirror verbose frame logs to stdout while writing logs/<video_input>.log.",
    ),
) -> None:
    identity = _build_identity(store_path)
    pipeline = FacePipeline(embedder=FaceEmbedder(), identity=identity)
    output_path = annotate_output or default_annotate_output_path(source)
    log_fn, log_handle = build_log_writer(
        source=source,
        verbose=verbose,
        tee_logs=tee_logs,
    )
    try:
        run_face_stage(
            source=source,
            identity=identity,
            pipeline=pipeline,
            output_path=output_path,
            limit_frames=limit_frames,
            verbose=verbose,
            log_fn=log_fn,
            update_templates=update_templates,
            auto_enroll_unknown=auto_enroll_unknown,
        )
    finally:
        if log_handle is not None:
            log_handle.close()


@app.command()
def serve(
    config_path: Path = typer.Option(
        Path("config/person-id-pi.json"),
        "--config",
        help="Live-service JSON configuration.",
    ),
) -> None:
    """Run motion-triggered person identity and beer counting on Pi Camera."""
    from .beer_detector import FilteredBeerDetector
    from .beverage_config import BeverageDetectorConfig
    from .beverage_detector import YoloBeverageDetector
    from .beverage_store import BeverageStore
    from .camera import PiCameraSource
    from .live_config import LiveConfig
    from .live_service import LiveService, install_signal_handlers

    config = LiveConfig.load(config_path)
    identity = _build_identity(Path(config.storage.identity_store))
    raw_detector = YoloBeverageDetector(
        BeverageDetectorConfig(
            model_path=config.beer.model_path,
            conf_min=config.beer.confidence_min,
            allowed_labels=("cup", "bottle", "can"),
        )
    )
    beer_detector = FilteredBeerDetector(
        detector=raw_detector,
        can_aspect_ratio_min=config.beer.can_aspect_ratio_min,
        can_aspect_ratio_max=config.beer.can_aspect_ratio_max,
    )
    service = LiveService(
        config=config,
        source=PiCameraSource(config.camera),
        face_embedder=FaceEmbedder(),
        identity=identity,
        beer_detector=beer_detector,
        beer_store=BeverageStore(config.storage.beer_store),
    )
    install_signal_handlers(service)
    service.run()


@app.command("camera-test")
def camera_test(
    camera_index: int = typer.Option(0, "--camera-index", help="OpenCV camera index."),
    seconds: float = typer.Option(
        3.0, "--seconds", min=0.5, max=30.0, help="Capture duration."
    ),
) -> None:
    """Open a local webcam briefly and report captured frame statistics."""
    from .camera import OpenCVCameraSource
    from .live_config import CameraConfig

    try:
        source = OpenCVCameraSource(camera_index, CameraConfig())
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        typer.echo(
            "Check System Settings > Privacy & Security > Camera and allow "
            "camera access for the application hosting this process."
        )
        raise typer.Exit(code=1) from exc
    started = time.monotonic()
    frames = 0
    width = 0
    height = 0
    try:
        while time.monotonic() - started < seconds:
            ok, frame = source.read()
            if not ok or frame is None:
                continue
            frames += 1
            height, width = frame.shape[:2]
    finally:
        source.close()
    elapsed = max(0.001, time.monotonic() - started)
    if frames == 0:
        typer.secho("Camera opened but returned no frames", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho(
        f"camera_ok index={camera_index} frames={frames} "
        f"resolution={width}x{height} measured_fps={frames / elapsed:.1f}",
        fg=typer.colors.GREEN,
    )


@app.command("smoke-test")
def smoke_test(
    seconds: float = typer.Option(
        60.0, "--seconds", min=10.0, max=300.0, help="Smoke-test duration."
    ),
    camera_index: int = typer.Option(0, "--camera-index", help="OpenCV camera index."),
    config_path: Path = typer.Option(
        Path("config/person-id-pi.json"), "--config", help="Base live configuration."
    ),
    output_dir: Path = typer.Option(
        Path("smoke"), "--output-dir", help="Isolated smoke-test output directory."
    ),
) -> None:
    """Run the live identity and beer pipeline against a local webcam."""
    from .beer_detector import FilteredBeerDetector
    from .beverage_config import BeverageDetectorConfig
    from .beverage_detector import YoloBeverageDetector
    from .beverage_store import BeverageStore
    from .camera import OpenCVCameraSource
    from .live_config import LiveConfig, StorageConfig
    from .live_service import LiveService

    config = LiveConfig.load(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = replace(
        config,
        storage=StorageConfig(
            identity_store=str(output_dir / "face_templates.json"),
            beer_store=str(output_dir / "beverage_events.json"),
            evidence_dir=str(output_dir / "events"),
            decision_log=str(output_dir / "decisions.jsonl"),
            record_evidence=True,
        ),
    )
    typer.echo("Loading face and beer models...")
    identity = _build_identity(Path(config.storage.identity_store))
    raw_detector = YoloBeverageDetector(
        BeverageDetectorConfig(
            model_path=config.beer.model_path,
            conf_min=config.beer.confidence_min,
            allowed_labels=("cup", "bottle", "can"),
        )
    )
    face_embedder = FaceEmbedder()
    try:
        source = OpenCVCameraSource(camera_index, config.camera)
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    service = LiveService(
        config=config,
        source=source,
        face_embedder=face_embedder,
        identity=identity,
        beer_detector=FilteredBeerDetector(
            raw_detector,
            can_aspect_ratio_min=config.beer.can_aspect_ratio_min,
            can_aspect_ratio_max=config.beer.can_aspect_ratio_max,
        ),
        beer_store=BeverageStore(config.storage.beer_store),
    )
    started = time.monotonic()
    outcomes = []
    typer.secho(
        f"smoke_test_started duration={seconds:.0f}s output={output_dir}",
        fg=typer.colors.BLUE,
    )
    try:
        while time.monotonic() - started < seconds:
            ok, frame = source.read()
            if not ok or frame is None:
                continue
            outcome = service.process_frame(frame, time.monotonic())
            if outcome is not None:
                outcomes.append(outcome)
    finally:
        final_outcome = service.close()
        if final_outcome is not None:
            outcomes.append(final_outcome)

    totals = service.beer_store.total_beers_by_user()
    typer.secho(
        f"smoke_test_complete events={len(outcomes)} users={identity.store.list_users()} "
        f"beer_totals={totals} decisions={config.storage.decision_log}",
        fg=typer.colors.GREEN,
    )


@app.command()
def enroll(
    user_id: str = typer.Argument(..., help="User ID to enroll."),
    source: str = typer.Argument(..., help="Path to video file or camera index."),
    store_path: Path = typer.Option(
        Path("profiles/face_templates.json"), "--store", help="Template store path."
    ),
    limit_frames: Optional[int] = typer.Option(
        None, "--limit-frames", help="Stop after N frames for quick checks."
    ),
    verbose: bool = typer.Option(
        True, "--verbose/--quiet", help="Print per-frame processing updates."
    ),
) -> None:
    identity = _build_identity(store_path)
    pipeline = FacePipeline(embedder=FaceEmbedder(), identity=identity)
    tracklet = pipeline.extract_primary_tracklet_from_video(
        source=source, limit_frames=limit_frames, verbose=verbose
    )
    if tracklet.n_used < identity.config.n_min:
        typer.secho(
            f"Enrollment failed: only {tracklet.n_used} samples (min={identity.config.n_min})",
            fg=typer.colors.YELLOW,
        )
        return
    updated = identity.update_templates(user_id, tracklet)
    if updated:
        typer.secho(
            f"Enrolled {user_id}: n_used={tracklet.n_used} dispersion={tracklet.dispersion:.3f}",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(
            f"Enrollment skipped: template too similar for {user_id}",
            fg=typer.colors.YELLOW,
        )


@app.command()
def list_users(
    store_path: Path = typer.Option(
        Path("profiles/face_templates.json"), "--store", help="Template store path."
    ),
) -> None:
    store = IdentityStore(store_path)
    for user_id in store.list_users():
        typer.echo(user_id)


@app.command("review-users")
def review_users(
    data_dir: Path = typer.Option(
        Path("smoke"), "--data-dir", help="Smoke or deployment data directory."
    ),
) -> None:
    """List generated identities with labels, beer totals, and evidence clips."""
    from .beverage_store import BeverageStore

    identity_path, beer_path, decisions_path = _review_data_paths(data_dir)
    if not identity_path.exists():
        typer.secho(f"Identity store not found: {identity_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    identity_store = IdentityStore(identity_path)
    beer_totals = BeverageStore(beer_path).total_beers_by_user()
    evidence_by_user: dict[str, list[str]] = {}
    if decisions_path.exists():
        for line in decisions_path.read_text(encoding="utf-8").splitlines():
            try:
                decision = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_id = decision.get("user_id")
            evidence_path = decision.get("evidence_path")
            if user_id and evidence_path:
                evidence_by_user.setdefault(str(user_id), []).append(str(evidence_path))

    for user_id in identity_store.list_users():
        name = identity_store.get_display_name(user_id) or "<unlabeled>"
        typer.secho(
            f"{user_id} name={name} templates={identity_store.template_count(user_id)} "
            f"beers={beer_totals.get(user_id, 0)}",
            fg=typer.colors.CYAN,
        )
        for evidence_path in evidence_by_user.get(user_id, []):
            typer.echo(f"  {evidence_path}")


@app.command("label-user")
def label_user(
    user_id: str = typer.Argument(..., help="Stable generated user ID."),
    display_name: str = typer.Argument(..., help="Human-readable display name."),
    data_dir: Path = typer.Option(
        Path("smoke"), "--data-dir", help="Smoke or deployment data directory."
    ),
) -> None:
    """Attach a display name to a stable identity without rewriting history."""
    identity_path, _, _ = _review_data_paths(data_dir)
    if not identity_path.exists():
        typer.secho(f"Identity store not found: {identity_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    store = IdentityStore(identity_path)
    if not store.has_user(user_id):
        typer.secho(f"User {user_id} not found", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not store.set_display_name(user_id, display_name):
        typer.secho(
            "Display name must be non-empty and unique across identities.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    typer.secho(
        f"Labeled {user_id} as {store.display_label(user_id)}. "
        "Future annotated videos will use this name.",
        fg=typer.colors.GREEN,
    )


@app.command()
def delete_user(
    user_id: str = typer.Argument(..., help="User ID to delete from the store."),
    store_path: Path = typer.Option(
        Path("profiles/face_templates.json"), "--store", help="Template store path."
    ),
) -> None:
    store = IdentityStore(store_path)
    if store.delete_user(user_id):
        typer.secho(f"Deleted {user_id}", fg=typer.colors.GREEN)
    else:
        typer.secho(f"User {user_id} not found", fg=typer.colors.YELLOW)


@app.command()
def rename_user(
    cur_user_id: str = typer.Argument(..., help="Current user ID to rename."),
    new_user_id: str = typer.Argument(..., help="New user ID."),
    store_path: Path = typer.Option(
        Path("profiles/face_templates.json"), "--store", help="Template store path."
    ),
) -> None:
    store = IdentityStore(store_path)
    if not store.has_user(cur_user_id):
        typer.secho(f"User {cur_user_id} not found", fg=typer.colors.YELLOW)
        return
    if cur_user_id != new_user_id and store.has_user(new_user_id):
        typer.secho(f"User {new_user_id} already exists", fg=typer.colors.YELLOW)
        return
    renamed = store.rename_user(cur_user_id, new_user_id)
    if not renamed:
        typer.secho(
            f"Unable to rename {cur_user_id} to {new_user_id}", fg=typer.colors.YELLOW
        )
        return
    typer.secho(f"Renamed {cur_user_id} to {new_user_id}", fg=typer.colors.GREEN)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
