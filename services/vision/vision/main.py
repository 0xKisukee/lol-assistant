"""Ligne de commande du service vision.

    python -m vision run            capture + détection + enregistrement + WebSocket
    python -m vision calibrate      ajuster le rectangle de la minimap et le sauvegarder
    python -m vision view <dossier> visionneuse d'un enregistrement
    python -m vision eval <dossier> précision et rappel sur les frames annotées
    python -m vision detect <image> détection sur une capture (test rapide)
    python -m vision assets         pré-télécharger les portraits Data Dragon

Mode hors-jeu : `run --source <png|dossier> --champions A,B,…` traite des
captures sur disque sans partie en cours (tests, démonstration).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from vision import __version__
from vision.assets import Assets, make_templates
from vision.calib import (CALIBRATION_FILE, FIT_LANDMARKS, GameConfig, MinimapMapper, estimate_calibration,
                          fit_rect_from_points, get_calibration, load_game_cfg, save_calibration)
from vision.capture import Frame, ImageSource, ScreenSource
from vision.detectors.champions import ChampionDetector, DetectorConfig
from vision.liveclient import GameClock, GameInfo, LiveClient
from vision.models import Calibration, Champion, Manifest
from vision.recorder import DEFAULT_GAMES_DIR, Recorder, new_game_id
from vision.server import DetectionServer

log = logging.getLogger("vision")


# ---- utilitaires --------------------------------------------------------------

def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_champions_arg(spec: str) -> list[Champion]:
    """`--champions Ahri,Garen,...` : 10 noms, les cinq premiers ORDER ; `Ahri:CHAOS` force l'équipe ; `*` = moi."""
    out = []
    for i, token in enumerate(t.strip() for t in spec.split(",") if t.strip()):
        is_me = token.startswith("*")
        token = token.lstrip("*")
        name, _, team = token.partition(":")
        team = team.upper() or ("ORDER" if i < 5 else "CHAOS")
        out.append(Champion(id=name, name=name, team=team, isMe=is_me))  # type: ignore[arg-type]
    return out


def build_detector(assets: Assets, champions: list[Champion], calib: Calibration, config: DetectorConfig,
                   portraits_dir: Path | None = None) -> ChampionDetector:
    portraits = {}
    for c in champions:
        if portraits_dir is not None:
            img = cv2.imread(str(portraits_dir / f"{c.id}.png"), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Portrait absent : {portraits_dir / (c.id + '.png')}")
            c.id = c.id
        else:
            c.id = assets.resolve_id(c.id, c.name)
            img = assets.portrait(c.id)
        portraits[c.id] = img
    templates = make_templates(portraits, calib.iconDiameter, calib.iconZoom)
    teams = {c.id: c.team for c in champions}
    my_team = next((c.team for c in champions if c.isMe), None)
    me_label = next((c.id for c in champions if c.isMe), None)
    return ChampionDetector(templates, teams, my_team=my_team, config=config, me_label=me_label)


def detector_config_from_args(args) -> DetectorConfig:
    cfg = DetectorConfig()
    if getattr(args, "threshold", None) is not None:
        cfg.threshold = args.threshold
    if getattr(args, "low_threshold", None) is not None:
        cfg.low_threshold = args.low_threshold
    if getattr(args, "ring_mode", None):
        cfg.ring_mode = args.ring_mode
    if getattr(args, "no_ring", False):
        cfg.use_ring = False
    if getattr(args, "scales", None):
        cfg.scales = tuple(float(s) for s in args.scales.split(","))
    return cfg


# ---- run ----------------------------------------------------------------------

def cmd_run(args) -> int:
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    server = None
    if not args.no_ws:
        server = DetectionServer(port=args.ws_port).start()
    assets = Assets(offline=args.offline)
    game_cfg = load_game_cfg(args.game_cfg)
    config = detector_config_from_args(args)
    portraits_dir = Path(args.portraits) if args.portraits else None

    while not stop.is_set():
        # 1. Attendre une partie (ou utiliser la liste fournie en mode hors-jeu).
        clock = None
        if args.champions:
            champions = parse_champions_arg(args.champions)
            info = GameInfo(champions=champions, stats=None)  # type: ignore[arg-type]
            game_mode, map_name = "OFFLINE", None
        else:
            client = LiveClient()
            if server:
                server.broadcast({"type": "status", "state": "waiting"})
            info = client.wait_for_game(stop=stop)
            if info is None:
                break
            champions = info.champions
            game_mode, map_name = info.stats.gameMode, info.stats.mapName
            clock = GameClock(client).start()
        log.info("Partie : %s, %s", game_mode, ", ".join(f"{c.name}({c.team[0]}{'*' if c.isMe else ''})" for c in champions))

        # 2. Source d'images.
        if args.source:
            source = ImageSource(args.source, fps=args.fps, loop=args.loop, clock=clock.now if clock else None,
                                 realtime=args.realtime)
        else:
            region = None
            if args.region:
                x, y, w, h = (int(v) for v in args.region.split(","))
                region = {"left": x, "top": y, "width": w, "height": h}
            source = ScreenSource(fps=args.fps, monitor=args.monitor, region=region, window=args.window,
                                  clock=clock.now if clock else None)
        first = next(iter(source.frames()))
        width, height = first.size

        # 3. Calibration et détecteur.
        calib = get_calibration(width, height, game_cfg, Path(args.calibration))
        log.info("Capture %dx%d, minimap %s (%s), icône %d px", width, height, calib.minimap.model_dump(),
                 calib.source, calib.iconDiameter)
        if calib.source == "estimate":
            log.warning("Rectangle de la minimap estimé, non vérifié : lance `python -m vision calibrate` pour l'ajuster.")
        mapper = MinimapMapper(calib)
        try:
            detector = build_detector(assets, champions, calib, config, portraits_dir)
        except Exception as e:
            log.error("Impossible de préparer les templates : %s", e)
            return 1

        # 4. Enregistrement.
        game_id = new_game_id()
        manifest = Manifest(
            gameId=game_id, startedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            patch=None if portraits_dir else assets.version, gameMode=game_mode, mapName=map_name,
            resolution=(width, height), calibration=calib, champions=champions, fps=args.fps,
            source=str(args.source) if args.source else "screen", detector=detector.describe(),
        )
        recorder = None if args.no_record else Recorder(manifest, Path(args.games_dir))
        if server:
            server.broadcast({"type": "game", "gameId": game_id, "patch": manifest.patch, "gameMode": game_mode,
                              "champions": [c.model_dump() for c in champions],
                              "calibration": calib.model_dump()})
            server.broadcast({"type": "status", "state": "running", "gameId": game_id})

        # 5. Boucle de traitement.
        def process(frame: Frame) -> None:
            t0 = time.perf_counter()
            crop = mapper.crop(frame.image)
            dets = detector.detect(crop, calib, frame.t, frame.id)
            latency = (time.perf_counter() - t0) * 1000.0
            if recorder:
                recorder.record(frame.id, frame.t, frame.wall, crop, dets, latency)
            if server:
                server.broadcast({"type": "detections", "gameId": game_id, "frameId": frame.id, "t": round(frame.t, 2),
                                  "latencyMs": round(latency, 1), "detections": [d.model_dump() for d in dets]})
            if frame.index % max(1, int(args.fps * 5)) == 0:
                seen = " ".join(f"{d.label}" for d in dets)
                log.info("frame %s t=%7.1fs %2d champions %5.0f ms  [%s]%s", frame.id, frame.t, len(dets), latency, seen,
                         f"  ws:{server.clients}" if server else "")

        process(first)

        def should_stop() -> bool:
            return stop.is_set() or (clock is not None and not clock.alive)

        try:
            for frame in source.frames(stop=should_stop):
                if frame.index == 0:
                    continue  # déjà traitée
                process(frame)
        finally:
            source.close()
            if clock:
                clock.stop()
            if recorder:
                recorder.close()
            if server:
                server.broadcast({"type": "status", "state": "ended", "gameId": game_id})
        if args.source or args.once:
            break
        log.info("Partie terminée.")
    if server:
        server.stop()
    return 0


# ---- calibrate ------------------------------------------------------------------

def cmd_calibrate(args) -> int:
    """Fenêtre interactive : la capture, le rectangle, les repères ; sauvegarde par résolution."""
    if args.image:
        frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if frame is None:
            log.error("Image illisible : %s", args.image)
            return 1
    else:
        src = ScreenSource(fps=1, monitor=args.monitor, window=args.window)
        log.info("Capture dans 2 s : mets le jeu au premier plan.")
        time.sleep(2.0)
        frame = src.grab().image
        src.close()
    height, width = frame.shape[:2]
    game_cfg = load_game_cfg(args.game_cfg)
    calib = get_calibration(width, height, game_cfg, Path(args.calibration))
    if args.reset:
        calib = estimate_calibration(width, height, game_cfg.minimap_scale, game_cfg.flip_minimap)
    rect = calib.minimap
    icon = calib.iconDiameter
    log.info("Capture %dx%d. Calibration %s : %s", width, height, calib.source, rect.model_dump())
    print(__doc_calibrate__)

    scale = min(1.0, 1400 / width, 900 / height)
    win_full, win_crop = "calibrate - capture", "calibrate - minimap (verifie nexus et tours)"
    cv2.namedWindow(win_full, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(win_crop, cv2.WINDOW_AUTOSIZE)

    state = {"rect": rect, "icon": icon, "fit": False, "points": [], "zoom": 1, "step": 1}

    def current_calib() -> Calibration:
        return Calibration(width=width, height=height, minimap=state["rect"], iconDiameter=state["icon"],
                           iconZoom=calib.iconZoom, minimapScale=calib.minimapScale, flipped=calib.flipped,
                           source="manual")

    def clamp() -> None:
        r = state["rect"]
        w = min(r.w, width); h = min(r.h, height)
        state["rect"] = r.model_copy(update={"w": w, "h": h, "x": max(0, min(width - w, r.x)),
                                             "y": max(0, min(height - h, r.y))})

    def add_point(fx: float, fy: float) -> None:
        if not state["fit"] or len(state["points"]) >= len(FIT_LANDMARKS):
            return
        name = FIT_LANDMARKS[len(state["points"])]
        state["points"].append((name, fx, fy))
        print(f"  {name} : ({fx:.0f}, {fy:.0f})")
        if len(state["points"]) == len(FIT_LANDMARKS):
            new_rect, residual = fit_rect_from_points(state["points"])
            state["rect"] = new_rect
            clamp()
            state["fit"] = False
            print(f"Rectangle ajusté : {state['rect'].model_dump()} (résidu {residual:.1f} px). "
                  f"Vérifie les croix, puis s pour sauver.")

    def on_mouse_full(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            add_point(x / scale, y / scale)

    def on_mouse_crop(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            r = state["rect"]
            add_point(r.x + x / state["zoom"], r.y + y / state["zoom"])

    cv2.setMouseCallback(win_full, on_mouse_full)
    cv2.setMouseCallback(win_crop, on_mouse_crop)

    def render() -> tuple[np.ndarray, np.ndarray]:
        r, ic = state["rect"], state["icon"]
        m = MinimapMapper(current_calib())
        full = frame.copy()
        cv2.rectangle(full, (r.x, r.y), (r.x2 - 1, r.y2 - 1), (0, 220, 255), 2)
        for name, px, py in state["points"]:
            cv2.drawMarker(full, (int(px), int(py)), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 18, 2)
        small = cv2.resize(full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if state["fit"]:
            todo = FIT_LANDMARKS[len(state["points"])]
            txt = f"AJUSTEMENT : clique sur  {todo}  ({len(state['points']) + 1}/{len(FIT_LANDMARKS)})   u: annuler  t: quitter"
            col = (0, 255, 0)
        else:
            txt = f"minimap x={r.x} y={r.y} cote={r.w}  icone={ic}px   o/p: icone -/+   t: ajuster par clics   s: sauver   q: quitter"
            col = (0, 220, 255)
        cv2.putText(small, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        crop = m.crop(frame)
        z = max(1, int(600 / max(r.w, 1)))
        state["zoom"] = z
        big = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST) if crop.size else np.zeros((100, 100, 3), np.uint8)
        for name, (px, py) in m.landmarks_crop().items():
            col = (235, 140, 40) if "ORDER" in name else (40, 40, 230)
            if state["fit"] and name == FIT_LANDMARKS[len(state["points"])]:
                col = (0, 255, 0)
            cv2.drawMarker(big, (int(px * z), int(py * z)), col, cv2.MARKER_CROSS, 16, 1)
            cv2.putText(big, name, (int(px * z) + 8, int(py * z) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        cv2.circle(big, (ic * z, ic * z), ic * z // 2, (0, 220, 255), 1)  # gabarit d'icône en haut à gauche
        return small, big

    while True:
        small, big = render()
        cv2.imshow(win_full, small)
        cv2.imshow(win_crop, big)
        key = cv2.waitKeyEx(30)
        if key == -1:
            if cv2.getWindowProperty(win_full, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        k = key & 0xFFFFFFFF
        r, step = state["rect"], state["step"]
        if k in (ord("q"), 27):
            break
        elif k == ord("s"):
            final = current_calib()
            save_calibration(final, Path(args.calibration))
            print(f"Calibration sauvegardée : {final.model_dump()}")
        elif k == ord("t"):
            state["fit"] = not state["fit"]
            state["points"] = []
            if state["fit"]:
                print("Ajustement par clics : clique sur chaque tour dans l'ordre affiché (dans l'une ou l'autre fenêtre).")
        elif k == ord("u") and state["points"]:
            state["points"].pop()
        elif k == ord("f"):
            state["step"] = 10 if step == 1 else 1
        elif k in (ord("j"), 63234, 65361, 2424832):
            state["rect"] = r.model_copy(update={"x": r.x - step})
        elif k in (ord("l"), 63235, 65363, 2555904):
            state["rect"] = r.model_copy(update={"x": r.x + step})
        elif k in (ord("i"), 63232, 65362, 2490368):
            state["rect"] = r.model_copy(update={"y": r.y - step})
        elif k in (ord("k"), 63233, 65364, 2621440):
            state["rect"] = r.model_copy(update={"y": r.y + step})
        elif k in (ord("+"), ord("=")):
            state["rect"] = r.model_copy(update={"w": r.w + step, "h": r.h + step})
        elif k in (ord("-"), ord("_")):
            state["rect"] = r.model_copy(update={"w": max(32, r.w - step), "h": max(32, r.h - step)})
        elif k in (ord("]"), ord("p")):
            state["icon"] = min(r.w // 2, state["icon"] + 1)
        elif k in (ord("["), ord("o")):
            state["icon"] = max(6, state["icon"] - 1)
        elif k == ord("c"):
            state["rect"] = r.model_copy(update={"x": 0 if calib.flipped else width - r.w, "y": height - r.h})
        clamp()
    cv2.destroyAllWindows()
    return 0


__doc_calibrate__ = """
Touches : t lance l'ajustement par clics (clique sur les quatre tours
extérieures dans l'ordre affiché, le rectangle est calculé), u annule le
dernier clic ; flèches ou i/j/k/l déplacent le rectangle, +/- changent sa
taille, o/p (ou [ ]) changent le diamètre d'icône, f alterne pas 1/10 px, c colle au
coin, s sauvegarde dans calibration.json, q quitte.
La fenêtre « minimap » affiche les repères attendus (nexus, tours
extérieures) : ajuste jusqu'à ce qu'ils tombent sur les icônes de la carte.
"""


# ---- view / eval / detect / assets ---------------------------------------------

def cmd_view(args) -> int:
    from vision.viewer import view
    view(Path(args.game_dir), zoom=args.zoom)
    return 0


def cmd_eval(args) -> int:
    from vision.eval import evaluate, write_report
    result = evaluate(Path(args.game_dir), Path(args.annotations) if args.annotations else None, args.tol)
    print(result.format())
    p = write_report(result, Path(args.game_dir))
    print(f"\nRapport écrit : {p}")
    return 0


def cmd_detect(args) -> int:
    """Détection sur une capture unique, résultat en JSON (et image annotée en option)."""
    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        log.error("Image illisible : %s", args.image)
        return 1
    height, width = frame.shape[:2]
    calib = get_calibration(width, height, load_game_cfg(args.game_cfg), Path(args.calibration))
    assets = Assets(offline=args.offline)
    champions = parse_champions_arg(args.champions)
    detector = build_detector(assets, champions, calib, detector_config_from_args(args),
                              Path(args.portraits) if args.portraits else None)
    mapper = MinimapMapper(calib)
    crop = mapper.crop(frame)
    dets = detector.detect(crop, calib, 0.0, "000000")
    print(f"{len(dets)} détections en {detector.last_latency_ms:.0f} ms (calibration {calib.source}, {calib.key})")
    for d in dets:
        print(d.model_dump_json())
    if args.out:
        z = 3
        big = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        for d in dets:
            c = (int(d.px.x * z), int(d.px.y * z))
            cv2.circle(big, c, calib.iconDiameter * z // 2, (0, 220, 255), 2)
            cv2.putText(big, f"{d.label} {d.confidence:.2f}", (c[0] - 20, c[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(args.out), big)
        print(f"Image annotée : {args.out}")
    return 0


def cmd_assets(args) -> int:
    assets = Assets(offline=args.offline)
    index = assets.champion_index()
    print(f"Data Dragon {assets.version} : {len(index)} champions")
    names = [n.strip() for n in args.champions.split(",")] if args.champions else sorted(index)
    for n in names:
        cid = assets.resolve_id(n, n)
        print(f"  {cid:16s} {assets.portrait_path(cid)}")
    return 0


# ---- parseur --------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vision", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"vision {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def common_capture(sp):
        sp.add_argument("--monitor", type=int, default=1, help="Moniteur mss à capturer (défaut 1)")
        sp.add_argument("--window", action="store_true", help="Capturer la fenêtre du jeu (Windows ; macOS avec pyobjc)")
        sp.add_argument("--game-cfg", help="Chemin de game.cfg")
        sp.add_argument("--calibration", default=str(CALIBRATION_FILE), help="Fichier calibration.json")

    def common_detector(sp):
        sp.add_argument("--champions", help="Mode hors-jeu : 10 champions séparés par des virgules (5 ORDER puis 5 CHAOS, *=moi)")
        sp.add_argument("--portraits", help="Dossier de portraits <Id>.png à utiliser à la place de Data Dragon")
        sp.add_argument("--offline", action="store_true", help="Ne pas interroger Data Dragon (cache seulement)")
        sp.add_argument("--threshold", type=float, help="Seuil de score d'une détection neuve (défaut 0.55)")
        sp.add_argument("--low-threshold", type=float, help="Seuil près de la dernière position connue (défaut 0.42)")
        sp.add_argument("--scales", help="Échelles autour du diamètre calibré, ex. 0.9,1.0,1.1")
        sp.add_argument("--ring-mode", choices=["ally", "side"], help="bleu = allié (défaut) ou bleu = ORDER")
        sp.add_argument("--no-ring", action="store_true", help="Ignorer la couleur de l'anneau")

    r = sub.add_parser("run", help="capture + détection + enregistrement + WebSocket")
    common_capture(r)
    common_detector(r)
    r.add_argument("--fps", type=float, default=2.0)
    r.add_argument("--region", help="Zone de capture x,y,w,h (points écran)")
    r.add_argument("--source", help="PNG ou dossier de PNG à la place de l'écran (mode hors-jeu)")
    r.add_argument("--loop", action="store_true", help="Avec --source : boucler")
    r.add_argument("--realtime", action="store_true", help="Avec --source : respecter --fps")
    r.add_argument("--games-dir", default=str(DEFAULT_GAMES_DIR))
    r.add_argument("--no-record", action="store_true")
    r.add_argument("--no-ws", action="store_true")
    r.add_argument("--ws-port", type=int, default=8765)
    r.add_argument("--once", action="store_true", help="S'arrêter à la fin de la première partie")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("calibrate", help="ajuster le rectangle de la minimap")
    common_capture(c)
    c.add_argument("--image", help="Calibrer sur une capture PNG au lieu de l'écran")
    c.add_argument("--reset", action="store_true", help="Repartir de l'estimation game.cfg")
    c.set_defaults(func=cmd_calibrate)

    v = sub.add_parser("view", help="visionneuse d'un enregistrement")
    v.add_argument("game_dir")
    v.add_argument("--zoom", type=int, default=3)
    v.set_defaults(func=cmd_view)

    e = sub.add_parser("eval", help="précision et rappel sur les frames annotées")
    e.add_argument("game_dir")
    e.add_argument("--annotations", help="Fichier d'annotations (défaut : <dossier>/annotations.jsonl)")
    e.add_argument("--tol", type=float, help="Tolérance en pixels (défaut : demi-diamètre d'icône)")
    e.set_defaults(func=cmd_eval)

    d = sub.add_parser("detect", help="détection sur une capture")
    d.add_argument("image")
    common_capture(d)
    common_detector(d)
    d.add_argument("--out", help="Écrire le crop annoté dans ce fichier")
    d.set_defaults(func=cmd_detect)

    a = sub.add_parser("assets", help="pré-télécharger les portraits Data Dragon")
    a.add_argument("--champions", help="Liste de noms (défaut : tous)")
    a.add_argument("--offline", action="store_true")
    a.set_defaults(func=cmd_assets)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    if args.command == "detect" and not args.champions:
        print("`detect` exige --champions (10 noms).", file=sys.stderr)
        return 2
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
