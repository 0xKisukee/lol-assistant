"""Enregistrement d'une partie : crops WebP, `detections.jsonl`, `manifest.json`.

Arborescence (PLAN.md §5.8) :

    data/games/<horodatage>/
    ├── manifest.json        patch, résolution, calibration, champions, gameMode…
    ├── detections.jsonl     une ligne par frame : {frameId, t, wall, file, latencyMs, detections[]}
    ├── frames/000123.webp   crop de la minimap
    ├── review.json          (visionneuse) frames marquées fausses
    └── annotations.jsonl    (visionneuse / main) vérité terrain pour `eval`
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from vision.models import Detection, FrameRecord, Manifest

log = logging.getLogger(__name__)

DEFAULT_GAMES_DIR = Path(__file__).resolve().parent.parent / "data" / "games"


def new_game_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return now.strftime("%Y-%m-%dT%H-%M-%S")


class Recorder:
    def __init__(self, manifest: Manifest, games_dir: Path | str = DEFAULT_GAMES_DIR, webp_quality: int = 85):
        self.dir = Path(games_dir) / manifest.gameId
        self.frames_dir = self.dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.webp_quality = webp_quality
        self._jsonl = (self.dir / "detections.jsonl").open("a", encoding="utf-8")
        self.count = 0
        self.write_manifest()
        log.info("Enregistrement dans %s", self.dir)

    def write_manifest(self) -> None:
        (self.dir / "manifest.json").write_text(
            json.dumps(self.manifest.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    def record(self, frame_id: str, t: float, wall: float, crop: np.ndarray,
               detections: list[Detection], latency_ms: float = 0.0) -> FrameRecord:
        rel = f"frames/{frame_id}.webp"
        cv2.imwrite(str(self.dir / rel), crop, [cv2.IMWRITE_WEBP_QUALITY, self.webp_quality])
        rec = FrameRecord(frameId=frame_id, t=t, wall=wall, file=rel, latencyMs=round(latency_ms, 1),
                          detections=detections)
        self._jsonl.write(rec.model_dump_json() + "\n")
        self.count += 1
        if self.count % 20 == 0:
            self._jsonl.flush()
        return rec

    def close(self) -> None:
        self._jsonl.close()
        self.manifest.frames = self.count
        self.manifest.endedAt = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.write_manifest()
        log.info("Enregistrement terminé : %d frames dans %s", self.count, self.dir)


# ---- lecture --------------------------------------------------------------

def load_manifest(game_dir: Path | str) -> Manifest:
    return Manifest.model_validate_json((Path(game_dir) / "manifest.json").read_text(encoding="utf-8"))


def load_records(game_dir: Path | str) -> list[FrameRecord]:
    p = Path(game_dir) / "detections.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(FrameRecord.model_validate_json(line))
    return out


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
