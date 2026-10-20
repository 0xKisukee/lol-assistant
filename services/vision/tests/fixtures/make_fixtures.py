"""Génère les fixtures synthétiques des tests (déterministe, graine fixe).

- portraits/<Nom>.png : dix « portraits » 128×128 aux textures distinctes ;
- minimap_synthetic.png + minimap_synthetic.json : fond de minimap avec les
  dix icônes collées (anneau bleu/rouge) et la vérité terrain ;
- frame_1080p.png : capture 1920×1080 factice avec la minimap en bas à droite,
  pour `calibrate --image` et `run --source`.

Avec `--frames N --out DIR`, écrit aussi une séquence de N captures où les
icônes se déplacent, pour tester `run --source DIR` sans partie.

Lancer : uv run python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ["Aatrox", "Bard", "Corki", "Darius", "Ezreal", "Fiora", "Garen", "Hecarim", "Irelia", "Jinx"]
TEAMS = ["ORDER"] * 5 + ["CHAOS"] * 5
ICON = 24
MINIMAP_SIDE = 300
FRAME_W, FRAME_H = 1920, 1080
BLUE = (235, 140, 40)  # BGR
RED = (40, 40, 230)


def make_portrait(rng: np.random.Generator, hue: int) -> np.ndarray:
    """Texture colorée lissée + quelques formes, pour imiter un portrait."""
    base = rng.random((16, 16, 3)).astype(np.float32)
    img = cv2.resize(base, (128, 128), interpolation=cv2.INTER_CUBIC)
    hsv = np.zeros((128, 128, 3), np.uint8)
    hsv[..., 0] = (hue + (img[..., 0] * 40)).astype(np.uint8) % 180
    hsv[..., 1] = (120 + img[..., 1] * 120).astype(np.uint8)
    hsv[..., 2] = (60 + img[..., 2] * 180).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for _ in range(4):
        c = (int(rng.integers(20, 108)), int(rng.integers(20, 108)))
        col = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.circle(out, c, int(rng.integers(8, 30)), col, -1)
        else:
            cv2.rectangle(out, c, (c[0] + int(rng.integers(10, 40)), c[1] + int(rng.integers(10, 40))), col, -1)
    return out


def make_background(rng: np.random.Generator, side: int) -> np.ndarray:
    """Fond de minimap : vert sombre bruité, rivière, lanes, quelques icônes grises."""
    noise = rng.random((side // 6, side // 6, 3)).astype(np.float32)
    tex = cv2.resize(noise, (side, side), interpolation=cv2.INTER_LINEAR)
    bg = np.zeros((side, side, 3), np.uint8)
    bg[..., 0] = (25 + tex[..., 0] * 30).astype(np.uint8)
    bg[..., 1] = (60 + tex[..., 1] * 50).astype(np.uint8)
    bg[..., 2] = (30 + tex[..., 2] * 30).astype(np.uint8)
    # Rivière (diagonale haut-gauche → bas-droite) et lanes (bords + diagonale inverse).
    cv2.line(bg, (0, 0), (side, side), (120, 90, 40), 14)
    cv2.line(bg, (0, side), (side, 0), (70, 110, 90), 8)
    cv2.line(bg, (12, 12), (12, side - 12), (70, 110, 90), 8)
    cv2.line(bg, (12, side - 12), (side - 12, side - 12), (70, 110, 90), 8)
    cv2.line(bg, (12, 12), (side - 12, 12), (70, 110, 90), 8)
    cv2.line(bg, (side - 12, 12), (side - 12, side - 12), (70, 110, 90), 8)
    # Tours (distracteurs gris) sur les lanes.
    for p in [(20, 90), (55, 160), (90, 240), (150, 150), (200, 100), (240, 55), (110, 280), (280, 110)]:
        cv2.circle(bg, p, 5, (150, 150, 150), -1)
        cv2.circle(bg, p, 5, (40, 40, 40), 1)
    return bg


def paste_icon(canvas: np.ndarray, portrait: np.ndarray, cx: int, cy: int, diameter: int, ring: tuple,
               brightness: float = 1.0) -> None:
    """Colle le disque central du portrait, assombri/éclairci, avec un anneau de 2 px."""
    icon = cv2.resize(portrait, (diameter, diameter), interpolation=cv2.INTER_AREA)
    icon = np.clip(icon.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    r = diameter // 2
    mask = np.zeros((diameter, diameter), np.uint8)
    cv2.circle(mask, (r, r), r - 1, 255, -1)
    x0, y0 = cx - r, cy - r
    h, w = canvas.shape[:2]
    for y in range(diameter):
        for x in range(diameter):
            if mask[y, x] and 0 <= y0 + y < h and 0 <= x0 + x < w:
                canvas[y0 + y, x0 + x] = icon[y, x]
    cv2.circle(canvas, (cx, cy), r, ring, 2, lineType=cv2.LINE_AA)


def positions(side: int, step: int = 0) -> list[tuple[int, int]]:
    """Dix positions ; `step` les fait dériver le long d'une trajectoire simple."""
    base = [(40, 250), (70, 200), (150, 140), (200, 40), (250, 60),
            (260, 120), (230, 230), (120, 60), (90, 120), (180, 200)]
    out = []
    for i, (x, y) in enumerate(base):
        dx = int(round(3 * step * np.cos(0.4 * i)))
        dy = int(round(3 * step * np.sin(0.4 * i)))
        x = int(np.clip(x + dx, ICON, side - ICON))
        y = int(np.clip(y + dy, ICON, side - ICON))
        out.append((x, y))
    return out


def render_minimap(portraits: dict[str, np.ndarray], bg: np.ndarray, step: int = 0,
                   hidden: set[str] | None = None) -> tuple[np.ndarray, list[dict]]:
    canvas = bg.copy()
    truth = []
    for (x, y), name, team in zip(positions(bg.shape[0], step), NAMES, TEAMS):
        if hidden and name in hidden:
            continue
        paste_icon(canvas, portraits[name], x, y, ICON, BLUE if team == "ORDER" else RED,
                   brightness=0.85 + 0.3 * ((NAMES.index(name) % 5) / 5))
        truth.append({"label": name, "team": team, "px": x, "py": y})
    return canvas, truth


def render_frame(minimap: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Capture factice : fond de jeu sombre, un bandeau HUD, minimap collée en bas à droite."""
    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    frame[:] = (30, 45, 35)
    cv2.rectangle(frame, (500, FRAME_H - 150), (1420, FRAME_H), (20, 20, 20), -1)
    for i in range(6):
        cv2.rectangle(frame, (560 + i * 60, FRAME_H - 120), (600 + i * 60, FRAME_H - 80), (90, 90, 90), -1)
    side = minimap.shape[0]
    frame[FRAME_H - side:, FRAME_W - side:] = minimap
    cv2.rectangle(frame, (FRAME_W - side - 1, FRAME_H - side - 1), (FRAME_W - 1, FRAME_H - 1), (160, 140, 60), 1)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=0, help="Nombre de captures de la séquence animée")
    ap.add_argument("--out", type=Path, default=None, help="Dossier de sortie de la séquence")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    (HERE / "portraits").mkdir(exist_ok=True)
    portraits = {}
    for i, name in enumerate(NAMES):
        p = make_portrait(rng, hue=(i * 37) % 180)
        portraits[name] = p
        cv2.imwrite(str(HERE / "portraits" / f"{name}.png"), p)

    bg = make_background(rng, MINIMAP_SIDE)
    minimap, truth = render_minimap(portraits, bg)
    cv2.imwrite(str(HERE / "minimap_synthetic.png"), minimap)
    (HERE / "minimap_synthetic.json").write_text(json.dumps({
        "iconDiameter": ICON, "side": MINIMAP_SIDE, "myTeam": "ORDER",
        "champions": truth,
    }, indent=2), encoding="utf-8")
    cv2.imwrite(str(HERE / "frame_1080p.png"), render_frame(minimap, rng), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(str(HERE / "minimap_empty.png"), bg)
    print("fixtures écrites dans", HERE)

    if args.frames and args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        gt = []
        for step in range(args.frames):
            hidden = {NAMES[(step // 3) % len(NAMES)]} if step % 3 == 2 else set()  # un champion disparaît parfois
            mm, tr = render_minimap(portraits, bg, step=step, hidden=hidden)
            cv2.imwrite(str(args.out / f"frame_{step:04d}.png"), render_frame(mm, rng), [cv2.IMWRITE_PNG_COMPRESSION, 9])
            gt.append({"frameId": f"{step:06d}", "champions": [{"label": c["label"], "px": c["px"], "py": c["py"]} for c in tr]})
        (args.out / "truth.jsonl").write_text("\n".join(json.dumps(g) for g in gt) + "\n", encoding="utf-8")
        print(f"{args.frames} captures écrites dans {args.out}")


if __name__ == "__main__":
    main()
