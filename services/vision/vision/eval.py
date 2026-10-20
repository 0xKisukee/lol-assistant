"""Précision et rappel des détections de champions sur des frames annotées.

Annotations : `annotations.jsonl` dans le dossier de la partie (ou un fichier
passé en argument), une ligne par frame annotée :

    {"frameId": "000123", "champions": [{"label": "Ahri", "px": 142, "py": 88}, ...]}

`px`/`py` sont les pixels du centre de l'icône dans le crop (ce que place la
visionneuse) ; on accepte aussi `x`/`y` normalisés (0..1, y vers le nord).
Une frame annotée avec une liste vide signifie « aucun champion visible » :
toute détection y est un faux positif. Les frames non annotées sont ignorées.

Appariement : une détection est vraie si une annotation du même label est à
moins de `tol` pixels (défaut 0,5 diamètre d'icône), chaque annotation ne
pouvant être appariée qu'une fois.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from vision.calib import MinimapMapper
from vision.models import FrameRecord, Manifest, Pos
from vision.recorder import load_jsonl, load_manifest, load_records


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    errors: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def mean_error(self) -> float:
        return sum(self.errors) / len(self.errors) if self.errors else 0.0


@dataclass
class EvalResult:
    frames: int
    overall: Counts
    per_label: dict[str, Counts]
    tol_px: float
    errors: list[dict] = field(default_factory=list)  # {frameId, t, label, kind: FP|FN, px, py, confidence?}

    def format_errors(self, label: str | None = None) -> str:
        rows = [e for e in self.errors if label is None or e["label"] == label]
        if not rows:
            return "Aucune erreur" + (f" pour {label}" if label else "") + "."
        lines = [f"{'frame':>7s} {'t':>7s} {'champion':12s} {'type':>4s} {'x':>6s} {'y':>6s} {'conf.':>5s}   autres détections sur la frame"]
        for e in rows:
            conf = f"{e['confidence']:.2f}" if e.get("confidence") is not None else "    "
            lines.append(f"{e['frameId']:>7s} {e['t']:7.1f} {e['label']:12s} {e['kind']:>4s} {e['px']:6.0f} {e['py']:6.0f} {conf:>5s}   {e.get('others', '')}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        def c(x: Counts) -> dict:
            return {"tp": x.tp, "fp": x.fp, "fn": x.fn, "precision": round(x.precision, 4),
                    "recall": round(x.recall, 4), "f1": round(x.f1, 4), "meanErrorPx": round(x.mean_error, 2)}
        return {"frames": self.frames, "tolPx": self.tol_px, "overall": c(self.overall),
                "perLabel": {k: c(v) for k, v in sorted(self.per_label.items())}}

    def format(self) -> str:
        o = self.overall
        hint = ""
        if o.fn == 0 and o.fp > o.tp:
            hint = ("\nAttention : aucun raté et plus de faux positifs que de vrais : les annotations sont "
                    "probablement incomplètes (des détections justes sans marqueur en face).")
        lines = [
            f"Frames annotées : {self.frames}   tolérance : {self.tol_px:.1f} px",
            f"Global  TP={o.tp} FP={o.fp} FN={o.fn}  précision={o.precision:.3f}  rappel={o.recall:.3f}  "
            f"F1={o.f1:.3f}  erreur moy.={o.mean_error:.2f} px",
            "",
            f"{'Champion':14s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'préc.':>7s} {'rappel':>7s}",
        ]
        for label, c in sorted(self.per_label.items()):
            lines.append(f"{label:14s} {c.tp:4d} {c.fp:4d} {c.fn:4d} {c.precision:7.3f} {c.recall:7.3f}")
        return "\n".join(lines) + hint


def annotation_px(a: dict, mapper: MinimapMapper) -> tuple[float, float]:
    if "px" in a and "py" in a:
        return float(a["px"]), float(a["py"])
    if "x" in a and "y" in a:
        return mapper.map_to_crop(Pos(x=float(a["x"]), y=float(a["y"])))
    raise ValueError(f"Annotation sans position : {a}")


def match_frame(dets: list[tuple[str, float, float]], truth: list[tuple[str, float, float]],
                tol: float) -> tuple[list[tuple[str, float]], list[str], list[str]]:
    """Apparie détections et vérité par label et distance. Renvoie (TP[(label, err)], FP[label], FN[label])."""
    used = [False] * len(truth)
    tps: list[tuple[str, float]] = []
    fps: list[str] = []
    # Gloutonnement par distance croissante, pour ne pas voler l'annotation la plus proche.
    pairs = []
    for i, (dl, dx, dy) in enumerate(dets):
        for j, (tl, tx, ty) in enumerate(truth):
            if dl == tl:
                dist = math.hypot(dx - tx, dy - ty)
                if dist <= tol:
                    pairs.append((dist, i, j))
    pairs.sort()
    matched_det = [False] * len(dets)
    for dist, i, j in pairs:
        if matched_det[i] or used[j]:
            continue
        matched_det[i] = used[j] = True
        tps.append((dets[i][0], dist))
    for i, d in enumerate(dets):
        if not matched_det[i]:
            fps.append(d[0])
    fns = [truth[j][0] for j in range(len(truth)) if not used[j]]
    return tps, fps, fns


def evaluate(game_dir: Path | str, annotations: Path | None = None, tol: float | None = None,
             records: list[FrameRecord] | None = None, manifest: Manifest | None = None,
             ignore: set[str] | None = None) -> EvalResult:
    """`ignore` : labels exclus des deux côtés (annotations et détections), ex. le joueur local."""
    game_dir = Path(game_dir)
    manifest = manifest or load_manifest(game_dir)
    records = records if records is not None else load_records(game_dir)
    ann_rows = load_jsonl(annotations or game_dir / "annotations.jsonl")
    if not ann_rows:
        raise FileNotFoundError(f"Aucune annotation ({annotations or game_dir / 'annotations.jsonl'}).")
    mapper = MinimapMapper(manifest.calibration)
    tol_px = tol if tol is not None else 0.5 * manifest.calibration.iconDiameter
    by_id = {r.frameId: r for r in records}
    overall, per_label = Counts(), defaultdict(Counts)
    errors: list[dict] = []
    frames = 0
    for row in ann_rows:
        rec = by_id.get(row["frameId"])
        if rec is None:
            continue
        frames += 1
        truth = [(a["label"], *annotation_px(a, mapper)) for a in row.get("champions", [])
                 if not ignore or a["label"] not in ignore]
        dets = [(d.label, d.px.x, d.px.y) for d in rec.detections
                if d.kind == "champion" and d.px is not None and not (ignore and d.label in ignore)]
        tps, fps, fns = match_frame(dets, truth, tol_px)
        # Détail des erreurs (positions) pour le diagnostic.
        matched = {(l, round(x), round(y)) for l, x, y in dets} - set()
        det_by_label = defaultdict(list)
        for d in rec.detections:
            if d.kind == "champion" and d.px is not None:
                det_by_label[d.label].append(d)
        others = " ".join(f"{d.label}({d.px.x:.0f},{d.px.y:.0f})" for d in rec.detections if d.px is not None)
        fp_labels = list(fps)
        for d in rec.detections:
            if d.px is None or d.label not in fp_labels:
                continue
            # Une détection est un FP si aucune annotation du même label n'est à moins de tol.
            if not any(l == d.label and math.hypot(d.px.x - x, d.px.y - y) <= tol_px for l, x, y in truth):
                errors.append({"frameId": rec.frameId, "t": rec.t, "label": d.label, "kind": "FP",
                               "px": d.px.x, "py": d.px.y, "confidence": d.confidence, "others": others})
                fp_labels.remove(d.label)
        fn_labels = list(fns)
        for l, x, y in truth:
            if l in fn_labels and not any(dl == l and math.hypot(dx - x, dy - y) <= tol_px for dl, dx, dy in dets):
                errors.append({"frameId": rec.frameId, "t": rec.t, "label": l, "kind": "FN", "px": x, "py": y,
                               "confidence": None, "others": others})
                fn_labels.remove(l)
        for label, err in tps:
            overall.tp += 1; overall.errors.append(err)
            per_label[label].tp += 1; per_label[label].errors.append(err)
        for label in fps:
            overall.fp += 1; per_label[label].fp += 1
        for label in fns:
            overall.fn += 1; per_label[label].fn += 1
    return EvalResult(frames=frames, overall=overall, per_label=dict(per_label), tol_px=tol_px, errors=errors)


def write_report(result: EvalResult, game_dir: Path) -> Path:
    p = Path(game_dir) / "eval.json"
    p.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return p
