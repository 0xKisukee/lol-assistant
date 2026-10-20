"""Visionneuse d'un dossier enregistré, image par image, avec annotation.

Fenêtre OpenCV : le crop de la minimap agrandi, les détections dessinées
(cercle couleur équipe, label, confiance), la frame courante et son gameTime.

Touches :
    → / d / espace  frame suivante          ← / a   frame précédente
    ↑ / w           +10 frames              ↓ / s   −10 frames
    p               lecture automatique (on/off)
    x               marquer / démarquer la frame comme fausse (review.json)
    e               mode annotation (on/off)
    1..9, 0         (annotation) choisir le champion 1..10
    clic gauche     (annotation) placer / déplacer le champion choisi
    clic droit      (annotation) supprimer le marqueur le plus proche
    c               (annotation) copier les détections comme annotation de la frame
    v               (annotation) valider la frame (écrite dans annotations.jsonl), même vide
    Suppr / Retour  (annotation) effacer l'annotation de la frame
    g               aller à une frame (saisie dans le terminal)
    q / Échap       quitter (sauvegarde)

`review.json` : {"wrong": [frameId…]} ; `annotations.jsonl` : voir eval.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from vision.models import FrameRecord, Manifest
from vision.recorder import dump_jsonl, load_jsonl, load_manifest, load_records

log = logging.getLogger(__name__)

BLUE, RED, WHITE, YELLOW, GRAY = (235, 140, 40), (40, 40, 230), (255, 255, 255), (0, 220, 255), (120, 120, 120)
KEYS_NEXT = {ord("d"), ord(" "), 3, 63235, 65363, 2555904}
KEYS_PREV = {ord("a"), 2, 63234, 65361, 2424832}
KEYS_UP = {ord("w"), 0, 63232, 65362, 2490368}
KEYS_DOWN = {ord("s"), 1, 63233, 65364, 2621440}


class Viewer:
    def __init__(self, game_dir: Path | str, zoom: int = 3):
        self.dir = Path(game_dir)
        self.manifest: Manifest = load_manifest(self.dir)
        self.records: list[FrameRecord] = load_records(self.dir)
        if not self.records:
            raise FileNotFoundError(f"Aucune frame dans {self.dir}")
        self.zoom = zoom
        self.index = 0
        self.playing = False
        self.annotating = False
        self.selected = 0
        self.labels = [c.id for c in self.manifest.champions]
        self.teams = {c.id: c.team for c in self.manifest.champions}
        self.my_team = next((c.team for c in self.manifest.champions if c.isMe), None)
        review_path = self.dir / "review.json"
        self.wrong: set[str] = set(json.loads(review_path.read_text()).get("wrong", [])) if review_path.exists() else set()
        self.annotations: dict[str, list[dict]] = {
            r["frameId"]: r.get("champions", []) for r in load_jsonl(self.dir / "annotations.jsonl")}
        self.draft: dict[str, list[dict]] = {}  # annotations en cours, non validées
        self._dirty = False
        self.window = f"vision — {self.dir.name}"

    # ---- persistance ------------------------------------------------------

    def save(self) -> None:
        (self.dir / "review.json").write_text(json.dumps({"wrong": sorted(self.wrong)}, indent=2), encoding="utf-8")
        rows = [{"frameId": fid, "champions": champs} for fid, champs in sorted(self.annotations.items())]
        dump_jsonl(self.dir / "annotations.jsonl", rows)
        self._dirty = False
        log.info("Sauvegardé : %d frames fausses, %d frames annotées.", len(self.wrong), len(self.annotations))

    # ---- rendu --------------------------------------------------------------

    def _color(self, label: str) -> tuple:
        team = self.teams.get(label)
        if team is None:
            return WHITE
        if self.my_team is not None:
            return BLUE if team == self.my_team else RED
        return BLUE if team == "ORDER" else RED

    def current_annotation(self) -> list[dict]:
        fid = self.records[self.index].frameId
        if fid in self.draft:
            return self.draft[fid]
        return self.annotations.get(fid, [])

    def render(self) -> np.ndarray:
        rec = self.records[self.index]
        crop = cv2.imread(str(self.dir / rec.file), cv2.IMREAD_COLOR)
        if crop is None:
            crop = np.zeros((self.manifest.calibration.minimap.h, self.manifest.calibration.minimap.w, 3), np.uint8)
        z = self.zoom
        img = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        r = self.manifest.calibration.iconDiameter * z // 2
        for d in rec.detections:
            if d.px is None:
                continue
            c = (int(d.px.x * z), int(d.px.y * z))
            cv2.circle(img, c, r, self._color(d.label), 2)
            cv2.putText(img, f"{d.label} {d.confidence:.2f}", (c[0] - r, c[1] - r - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
        if self.annotating:
            for a in self.current_annotation():
                c = (int(a["px"] * z), int(a["py"] * z))
                cv2.drawMarker(img, c, YELLOW, cv2.MARKER_CROSS, 2 * r, 1)
                cv2.putText(img, a["label"], (c[0] + r, c[1] + r), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)
        # Bandeau d'état.
        h = 72
        band = np.zeros((h, img.shape[1], 3), np.uint8)
        fid = rec.frameId
        status = f"{self.index + 1}/{len(self.records)}  frame {fid}  t={rec.t:7.1f}s  {len(rec.detections)} det  {rec.latencyMs:.0f} ms"
        if fid in self.wrong:
            status += "  [FAUSSE]"
        if fid in self.annotations:
            status += "  [annotee]"
        if fid in self.draft:
            status += "  [brouillon]"
        cv2.putText(band, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        if self.annotating:
            sel = self.labels[self.selected] if self.labels else "?"
            cv2.putText(band, f"ANNOTATION  champion [{self.selected + 1}] {sel}   clic=placer  droit=retirer  c=copier det  v=valider",
                        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)
        else:
            cv2.putText(band, "d/a: +-1  w/s: +-10  p: lecture  x: fausse  e: annoter  q: quitter",
                        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1, cv2.LINE_AA)
        names = "  ".join(f"{i + 1 if i < 9 else 0}:{l}" for i, l in enumerate(self.labels))
        cv2.putText(band, names, (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1, cv2.LINE_AA)
        return np.vstack([img, band])

    # ---- interactions -----------------------------------------------------

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if not self.annotating or not self.labels:
            return
        px, py = x / self.zoom, y / self.zoom
        rec = self.records[self.index]
        if py > self.manifest.calibration.minimap.h:
            return
        fid = rec.frameId
        current = list(self.current_annotation())
        if event == cv2.EVENT_LBUTTONDOWN:
            label = self.labels[self.selected]
            current = [a for a in current if a["label"] != label]
            current.append({"label": label, "px": round(px, 1), "py": round(py, 1)})
            self.draft[fid] = current
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            nearest = min(current, key=lambda a: (a["px"] - px) ** 2 + (a["py"] - py) ** 2)
            current.remove(nearest)
            self.draft[fid] = current

    def _goto(self, index: int) -> None:
        self.index = max(0, min(len(self.records) - 1, index))

    def handle_key(self, key: int) -> bool:
        """Traite une touche ; renvoie False pour quitter."""
        fid = self.records[self.index].frameId
        if key in (ord("q"), 27):
            return False
        if key in KEYS_NEXT:
            self._goto(self.index + 1)
        elif key in KEYS_PREV:
            self._goto(self.index - 1)
        elif key in KEYS_UP:
            self._goto(self.index + 10)
        elif key in KEYS_DOWN:
            self._goto(self.index - 10)
        elif key == ord("p"):
            self.playing = not self.playing
        elif key == ord("x"):
            self.wrong.symmetric_difference_update({fid})
            self._dirty = True
        elif key == ord("e"):
            self.annotating = not self.annotating
        elif key == ord("g"):
            try:
                self._goto(int(input("Aller à la frame (index 1..N) : ")) - 1)
            except ValueError:
                pass
        elif self.annotating:
            if ord("0") <= key <= ord("9"):
                n = (key - ord("0") - 1) % 10
                if n < len(self.labels):
                    self.selected = n
            elif key == ord("c"):
                rec = self.records[self.index]
                self.draft[fid] = [{"label": d.label, "px": d.px.x, "py": d.px.y} for d in rec.detections if d.px]
            elif key == ord("v"):
                self.annotations[fid] = self.draft.pop(fid, self.current_annotation())
                self._dirty = True
                self._goto(self.index + 1)
            elif key in (8, 127, 3014656):
                self.draft.pop(fid, None)
                self.annotations.pop(fid, None)
                self._dirty = True
        return True

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self._on_mouse)
        try:
            while True:
                cv2.imshow(self.window, self.render())
                delay = max(1, int(1000 / max(self.manifest.fps, 0.5))) if self.playing else 30
                key = cv2.waitKeyEx(delay)
                if key == -1:
                    if self.playing:
                        if self.index >= len(self.records) - 1:
                            self.playing = False
                        else:
                            self._goto(self.index + 1)
                    if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    continue
                if not self.handle_key(key & 0xFFFFFFFF if key > 0x10FFFF else key):
                    break
        finally:
            cv2.destroyAllWindows()
            self.save()


def view(game_dir: Path | str, zoom: int = 3) -> None:
    Viewer(game_dir, zoom=zoom).run()
