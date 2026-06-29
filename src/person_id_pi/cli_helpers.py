from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TextIO

import typer

from .face_pipeline import FacePipeline, FrameTrackAnnotation
from .face_types import IdentityDecision
from .identity_engine import IdentityEngine


@dataclass(frozen=True)
class FaceStageResult:
    frame_annotations: list[list[FrameTrackAnnotation]]
    decisions_by_track: dict[int, IdentityDecision]


def default_annotate_output_path(source: str) -> Path:
    src = Path(source)
    stem = src.stem if src.suffix else src.name
    return Path("runs") / f"{stem}_annotated.mp4"


def default_log_path(source: str) -> Path:
    src = Path(source)
    stem = src.stem if src.suffix else src.name
    return Path("logs") / f"{stem}.log"


def build_log_writer(
    source: str,
    verbose: bool,
    tee_logs: bool,
) -> tuple[Optional[Callable[[str], None]], Optional[TextIO]]:
    if not verbose:
        return None, None
    log_path = default_log_path(source)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    typer.secho(f"Verbose logs written to {log_path}", fg=typer.colors.BLUE)

    def _file_only(message: str) -> None:
        log_handle.write(f"{message}\n")

    def _tee(message: str) -> None:
        log_handle.write(f"{message}\n")
        print(message)

    return (_tee if tee_logs else _file_only), log_handle


def run_face_stage(
    *,
    source: str,
    identity: IdentityEngine,
    pipeline: FacePipeline,
    output_path: Path,
    limit_frames: Optional[int],
    verbose: bool,
    log_fn: Optional[Callable[[str], None]],
    update_templates: bool,
    auto_enroll_unknown: bool,
) -> FaceStageResult:
    tracklets_by_id, frame_annotations = (
        pipeline.extract_tracklets_with_annotations_from_video(
            source=source,
            limit_frames=limit_frames,
            verbose=verbose,
            log_fn=log_fn,
        )
    )
    decisions_by_track: dict[int, IdentityDecision] = {}
    for track_id in sorted(tracklets_by_id.keys()):
        tracklet = tracklets_by_id[track_id]
        decision = identity.match(tracklet)
        auto_enroll_status: Optional[str] = None
        if (
            update_templates
            and decision.user_id
            and identity.should_update_templates(decision, tracklet)
        ):
            identity.update_templates(decision.user_id, tracklet)
        if auto_enroll_unknown:
            block_reason = identity.auto_enroll_block_reason(decision, tracklet)
            if block_reason is None:
                new_user_id = identity.store.generate_new_user_id()
                added = identity.update_templates(new_user_id, tracklet)
                if added:
                    decision = IdentityDecision(
                        user_id=new_user_id,
                        score=decision.score,
                        margin=decision.margin,
                        accepted=True,
                        reason="auto_enrolled_unknown",
                    )
                    auto_enroll_status = "enrolled"
                else:
                    auto_enroll_status = "rejected:template_too_similar"
            else:
                auto_enroll_status = f"rejected:{block_reason}"
        decisions_by_track[track_id] = decision
        fields = [
            f"track={track_id}",
            f"accepted={decision.accepted}",
            f"user_id={decision.user_id}",
            f"score={decision.score:.3f}",
            f"margin={decision.margin:.3f}",
            f"n_used={tracklet.n_used}",
            f"dispersion={tracklet.dispersion:.3f}",
            f"reason={decision.reason}",
        ]
        if auto_enroll_unknown:
            fields.append(f"auto_enroll={auto_enroll_status}")
        typer.secho(" ".join(fields), fg=typer.colors.CYAN)

    pipeline.write_multi_face_annotations(
        source=source,
        output_path=output_path,
        frame_annotations=frame_annotations,
        decisions=decisions_by_track,
    )
    typer.secho(f"Annotated video written to {output_path}", fg=typer.colors.GREEN)
    return FaceStageResult(
        frame_annotations=frame_annotations,
        decisions_by_track=decisions_by_track,
    )
