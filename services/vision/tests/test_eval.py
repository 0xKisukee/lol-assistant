"""Appariement et métriques de eval.py."""

import json
from pathlib import Path

from vision.eval import evaluate, match_frame
from vision.models import Calibration, Champion, Detection, FrameRecord, Manifest, MinimapRect, Pos


def test_match_frame_basic():
    dets = [("Ahri", 10, 10), ("Garen", 50, 50), ("Jinx", 90, 90)]
    truth = [("Ahri", 12, 11), ("Garen", 80, 80), ("Zed", 30, 30)]
    tps, fps, fns = match_frame(dets, truth, tol=6)
    assert [l for l, _ in tps] == ["Ahri"]
    assert sorted(fps) == ["Garen", "Jinx"]
    assert sorted(fns) == ["Garen", "Zed"]


def test_match_frame_one_to_one():
    dets = [("Ahri", 10, 10), ("Ahri", 13, 10)]
    truth = [("Ahri", 11, 10)]
    tps, fps, fns = match_frame(dets, truth, tol=6)
    assert len(tps) == 1 and tps[0][1] == 1.0 and fps == ["Ahri"] and fns == []


def test_evaluate_end_to_end(tmp_path: Path):
    calib = Calibration(width=300, height=300, minimap=MinimapRect(x=0, y=0, w=300, h=300), iconDiameter=24)
    manifest = Manifest(gameId="g", startedAt="now", resolution=(300, 300), calibration=calib,
                        champions=[Champion(id="Ahri", name="Ahri", team="ORDER"), Champion(id="Zed", name="Zed", team="CHAOS")],
                        fps=2.0)
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())

    def det(label, x, y, fid):
        return Detection(t=0, frameId=fid, label=label, confidence=0.9, px=Pos(x=x, y=y), pos=Pos(x=x / 300, y=1 - y / 300))

    records = [
        FrameRecord(frameId="000000", t=0, wall=0, file="frames/000000.webp", detections=[det("Ahri", 100, 100, "000000")]),
        FrameRecord(frameId="000001", t=1, wall=0, file="frames/000001.webp", detections=[det("Ahri", 100, 100, "000001"), det("Zed", 200, 200, "000001")]),
        FrameRecord(frameId="000002", t=2, wall=0, file="frames/000002.webp", detections=[det("Zed", 10, 10, "000002")]),
        FrameRecord(frameId="000003", t=3, wall=0, file="frames/000003.webp", detections=[]),  # non annotée
    ]
    (tmp_path / "detections.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in records))
    annotations = [
        {"frameId": "000000", "champions": [{"label": "Ahri", "px": 102, "py": 99}, {"label": "Zed", "px": 250, "py": 250}]},
        {"frameId": "000001", "champions": [{"label": "Ahri", "x": 100 / 300, "y": 1 - 100 / 300}, {"label": "Zed", "px": 201, "py": 200}]},
        {"frameId": "000002", "champions": []},
    ]
    (tmp_path / "annotations.jsonl").write_text("".join(json.dumps(a) + "\n" for a in annotations))

    r = evaluate(tmp_path)
    assert r.frames == 3 and r.tol_px == 12
    assert (r.overall.tp, r.overall.fp, r.overall.fn) == (3, 1, 1)
    assert r.per_label["Ahri"].recall == 1.0 and r.per_label["Zed"].precision == 0.5
    assert abs(r.overall.precision - 0.75) < 1e-9 and abs(r.overall.recall - 0.75) < 1e-9
    assert "précision=0.750" in r.format()
    assert r.to_dict()["overall"]["tp"] == 3
