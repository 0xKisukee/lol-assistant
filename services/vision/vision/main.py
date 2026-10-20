"""Ligne de commande du service vision.

    python -m vision run            capture + détection + enregistrement + WebSocket
    python -m vision calibrate      ajuster le rectangle de la minimap et le sauvegarder
    python -m vision view <dossier> visionneuse d'un enregistrement
    python -m vision eval <dossier> précision et rappel sur les frames annotées
    python -m vision detect <image> détection sur une capture (test rapide)
    python -m vision replay <dossier> rejouer le détecteur sur les crops enregistrés (autres seuils)
    python -m vision debug <dossier>  scores par champion (une frame ou statistiques)
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
                   portraits_dir: Path | None = None, icon_source: str = "circle") -> ChampionDetector:
    portraits = {}
    for c in champions:
        if portraits_dir is not None:
            img = cv2.imread(str(portraits_dir / f"{c.id}.png"), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Portrait absent : {portraits_dir / (c.id + '.png')}")
        else:
            c.id = assets.resolve_id(c.id, c.name)
            img = assets.icon(c.id, c.skinId, source=icon_source)
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
    if getattr(args, "no_static", False):
        cfg.static_frames = 0
    if getattr(args, "no_recover", False):
        cfg.recover_overlaps = False
    if getattr(args, "no_hidden", False):
        cfg.infer_hidden_allies = False
    if getattr(args, "no_ring_only", False):
        cfg.ring_only_recovery = False
    for item in getattr(args, "set", None) or []:
        key, _, value = item.partition("=")
        if not hasattr(cfg, key):
            raise SystemExit(f"--set : paramètre inconnu {key!r} (voir DetectorConfig)")
        current = getattr(cfg, key)
        if isinstance(current, bool):
            setattr(cfg, key, value.lower() in ("1", "true", "oui", "yes"))
        elif isinstance(current, int):
            setattr(cfg, key, int(value))
        elif isinstance(current, float):
            setattr(cfg, key, float(value))
        elif isinstance(current, tuple):
            setattr(cfg, key, tuple(float(v) for v in value.split(",")))
        else:
            setattr(cfg, key, value)
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
        if args.icon:
            calib = calib.model_copy(update={"iconDiameter": args.icon})
        if args.icon_zoom:
            calib = calib.model_copy(update={"iconZoom": args.icon_zoom})
        if args.crop_margin is not None:
            calib = calib.model_copy(update={"cropMargin": args.crop_margin})
        elif not calib.cropMargin:
            calib = calib.model_copy(update={"cropMargin": calib.iconDiameter})  # icônes de la fontaine au bord
        log.info("Capture %dx%d, minimap %s (%s), icône %d px, marge %d px", width, height, calib.minimap.model_dump(),
                 calib.source, calib.iconDiameter, calib.cropMargin)
        if calib.source == "estimate":
            log.warning("Rectangle de la minimap estimé, non vérifié : lance `python -m vision calibrate` pour l'ajuster.")
        mapper = MinimapMapper(calib)
        try:
            detector = build_detector(assets, champions, calib, config, portraits_dir, args.icon_source)
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
        name_to_id = {c.summonerName: c.id for c in champions if c.summonerName}

        def process(frame: Frame) -> None:
            t0 = time.perf_counter()
            crop = mapper.crop(frame.image)
            dead = {name_to_id[n] for n in clock.dead if n in name_to_id} if clock else set()
            dets = detector.detect(crop, calib, frame.t, frame.id, exclude=dead)
            latency = (time.perf_counter() - t0) * 1000.0
            if recorder:
                recorder.record(frame.id, frame.t, frame.wall, crop, dets, latency, dead=sorted(dead))
            if server:
                server.broadcast({"type": "detections", "gameId": game_id, "frameId": frame.id, "t": round(frame.t, 2),
                                  "latencyMs": round(latency, 1), "dead": sorted(dead),
                                  "detections": [d.model_dump() for d in dets]})
            if frame.index % max(1, int(args.fps * 5)) == 0:
                seen = " ".join(f"{d.label}" for d in dets)
                log.info("frame %s t=%7.1fs %2d champions %5.0f ms  [%s]%s%s", frame.id, frame.t, len(dets), latency, seen,
                         f"  morts: {' '.join(sorted(dead))}" if dead else "",
                         f"  ws:{server.clients}" if server else "")

        process(first)

        def should_stop() -> bool:
            return stop.is_set() or (clock is not None and not clock.alive)

        hidden = False
        skipped = 0
        try:
            for frame in source.frames(stop=should_stop):
                if frame.index == 0:
                    continue  # déjà traitée
                visible = source.game_visible() if hasattr(source, "game_visible") else None
                if visible is False:
                    # Jeu recouvert par une autre fenêtre (Alt+Tab) : on ne traite ni n'enregistre.
                    skipped += 1
                    if not hidden:
                        hidden = True
                        log.info("Jeu masqué par une autre fenêtre : capture en pause.")
                        if server:
                            server.broadcast({"type": "status", "state": "paused", "gameId": game_id})
                    continue
                if hidden:
                    hidden = False
                    log.info("Jeu de retour au premier plan (%d frames ignorées).", skipped)
                    if server:
                        server.broadcast({"type": "status", "state": "running", "gameId": game_id})
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
    if args.set_icon:
        from vision.calib import load_calibrations
        path = Path(args.calibration)
        all_ = load_calibrations(path)
        if not all_:
            log.error("Aucune calibration dans %s : lance d'abord `calibrate` normalement.", path)
            return 1
        keys = [args.resolution] if args.resolution else list(all_)
        for key in keys:
            if key not in all_:
                log.error("Résolution %s absente de %s (présentes : %s)", key, path, ", ".join(all_))
                return 1
            save_calibration(all_[key].model_copy(update={"iconDiameter": args.set_icon, "source": "manual"}), path)
            print(f"{key} : diamètre d'icône = {args.set_icon} px")
        return 0
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
    detections = args.detections
    if args.replay:
        # Rejouer le détecteur à jour sur tout l'enregistrement (le suivi dépend de l'ordre des
        # frames), puis ouvrir la visionneuse sur ce résultat.
        args.out = args.out or "detections.replay.jsonl"
        if not hasattr(args, "portraits"):
            args.portraits = None
        for name, default in (("offline", False), ("icon", None), ("icon_zoom", None), ("icon_source", "circle"),
                              ("threshold", None), ("low_threshold", None), ("scales", None), ("ring_mode", None),
                              ("no_ring", False), ("no_static", False), ("no_recover", False), ("no_hidden", False),
                              ("no_ring_only", False), ("set", None)):
            if not hasattr(args, name):
                setattr(args, name, default)
        print("Rejeu du détecteur sur l'enregistrement…")
        cmd_replay(args)
        detections = args.out
    view(Path(args.game_dir), zoom=args.zoom, detections=detections)
    return 0


def cmd_eval(args) -> int:
    from vision.eval import evaluate, write_report
    records = None
    if args.detections:
        from vision.models import FrameRecord
        p = Path(args.game_dir) / args.detections if not Path(args.detections).is_absolute() else Path(args.detections)
        records = [FrameRecord.model_validate_json(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"Détections : {p}")
    ignore = {x.strip() for x in args.ignore.split(",") if x.strip()} if args.ignore else None
    result = evaluate(Path(args.game_dir), Path(args.annotations) if args.annotations else None, args.tol,
                      records=records, ignore=ignore)
    print(result.format())
    if args.errors is not None:
        print()
        print(result.format_errors(args.errors or None))
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
    if args.icon:
        calib = calib.model_copy(update={"iconDiameter": args.icon})
    detector = build_detector(assets, champions, calib, detector_config_from_args(args),
                              Path(args.portraits) if args.portraits else None, args.icon_source)
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


def _detector_for_game(game_dir: Path, args) -> tuple:
    """Détecteur reconstruit depuis le manifest d'un enregistrement (champions, calibration, patch)."""
    from vision.recorder import load_manifest
    manifest = load_manifest(game_dir)
    calib = manifest.calibration
    if getattr(args, "icon", None):
        calib = calib.model_copy(update={"iconDiameter": args.icon})
    if getattr(args, "icon_zoom", None):
        calib = calib.model_copy(update={"iconZoom": args.icon_zoom})
    assets = Assets(version=manifest.patch, offline=args.offline)
    champions = [c.model_copy() for c in manifest.champions]
    detector = build_detector(assets, champions, calib, detector_config_from_args(args),
                              Path(args.portraits) if args.portraits else None, args.icon_source)
    return manifest, calib, detector


def cmd_replay(args) -> int:
    """Rejoue le détecteur sur les crops d'un enregistrement et écrit un nouveau detections.jsonl."""
    from vision.models import FrameRecord
    from vision.recorder import load_records
    game_dir = Path(args.game_dir)
    manifest, calib, detector = _detector_for_game(game_dir, args)
    records = load_records(game_dir)
    out = game_dir / (args.out or "detections.replay.jsonl")
    n_det, t_total = 0, 0.0
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            crop = cv2.imread(str(game_dir / rec.file), cv2.IMREAD_COLOR)
            if crop is None:
                continue
            t0 = time.perf_counter()
            dets = detector.detect(crop, calib, rec.t, rec.frameId, exclude=set(rec.dead))
            ms = (time.perf_counter() - t0) * 1000
            t_total += ms
            n_det += len(dets)
            f.write(FrameRecord(frameId=rec.frameId, t=rec.t, wall=rec.wall, file=rec.file, latencyMs=round(ms, 1),
                                dead=rec.dead, detections=dets).model_dump_json() + "\n")
    n = max(1, len(records))
    print(f"{len(records)} frames rejouées, {n_det} détections ({n_det / n:.1f} par frame), "
          f"{t_total / n:.0f} ms par frame, échelle verrouillée : {detector.locked_diameter} px")
    print(f"Écrit : {out}   (évaluer avec : eval {game_dir} --detections {out.name})")
    return 0


def _tune_icon(game_dir: Path, manifest, calib, records, args) -> int:
    """Pour chaque diamètre candidat : score brut moyen des alliés (toujours visibles) et rappel grossier."""
    from vision.detectors.champions import DetectorConfig
    assets = Assets(version=manifest.patch, offline=args.offline)
    allies_team = next((c.team for c in manifest.champions if c.isMe), None)
    allies = [c.id for c in manifest.champions if allies_team is None or c.team == allies_team]
    step = max(1, len(records) // max(1, min(args.max_frames, 40)))
    crops = []
    for rec in records[::step]:
        img = cv2.imread(str(game_dir / rec.file), cv2.IMREAD_COLOR)
        if img is not None:
            crops.append(img)
    base = calib.iconDiameter
    cands = sorted({max(8, int(round(base * f))) for f in (0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2)})
    zooms = (1.0, 1.1, 1.2, 1.3, 1.45) if args.zoom else (calib.iconZoom,)
    print(f"{len(crops)} frames, alliés {allies} (toujours visibles sur la minimap), diamètre calibré {base} px")
    print(f"{'diam':>5s} {'zoom':>5s} {'score moy alliés':>17s} {'alliés >= 0.5':>14s} {'alliés >= 0.6':>14s}   score moy par allié")
    best = None
    for d, zm in ((d, zm) for zm in zooms for d in cands):
        c2 = calib.model_copy(update={"iconDiameter": d, "iconZoom": zm})
        cfg = DetectorConfig(scales=(1.0,), use_ring=False, threshold=0.0, low_threshold=0.0,
                             scale_probe_frames=10 ** 9)
        det = build_detector(assets, [c.model_copy() for c in manifest.champions], c2, cfg,
                             Path(args.portraits) if args.portraits else None, args.icon_source)
        per = {a: [] for a in allies}
        for crop in crops:
            top = {}
            for c in det.candidates(crop):
                if c.label in per and c.score > top.get(c.label, -1):
                    top[c.label] = c.score
            for a in allies:
                per[a].append(top.get(a, 0.0))
        means = {a: sum(v) / len(v) for a, v in per.items() if v}
        avg = sum(means.values()) / len(means)
        r5 = sum(1 for v in per.values() for x in v if x >= 0.5) / max(1, sum(len(v) for v in per.values()))
        r6 = sum(1 for v in per.values() for x in v if x >= 0.6) / max(1, sum(len(v) for v in per.values()))
        detail = "  ".join(f"{a[:6]}={m:.2f}" for a, m in means.items())
        print(f"{d:5d} {zm:5.2f} {avg:17.3f} {r5:14.0%} {r6:14.0%}   {detail}")
        if best is None or avg > best[2]:
            best = (d, zm, avg)
    print(f"\nMeilleur : diamètre {best[0]} px, zoom {best[1]:.2f} (score moyen {best[2]:.3f}). "
          f"Reporte-les avec `--icon {best[0]}` et, si le zoom n'est pas 1, `--icon-zoom {best[1]}`.")
    if args.zoom:
        print("Note : diamètre ÷ zoom constant = même contenu comparé ; le diamètre doit rester celui de l'icône "
              "réelle (l'anneau est cherché sur son bord), seul le zoom change ce qui est comparé.")
    return 0


def _annotation_for(game_dir: Path, frame_id: str) -> list[dict] | None:
    """Annotation (vérité) de cette frame si elle existe dans annotations.jsonl."""
    from vision.recorder import load_jsonl
    for row in load_jsonl(game_dir / "annotations.jsonl"):
        if row.get("frameId") == frame_id:
            return row.get("champions", [])
    return None


def _masked_ncc_at(crop, tpl, cx: float, cy: float) -> float | None:
    """NCC masquée du template à une position donnée du crop (calcul direct)."""
    d = tpl.diameter
    x0, y0 = int(round(cx - d / 2)), int(round(cy - d / 2))
    if x0 < 0 or y0 < 0 or x0 + d > crop.shape[1] or y0 + d > crop.shape[0]:
        return None
    M = tpl.mask > 0
    w = crop[y0:y0 + d, x0:x0 + d].astype(np.float64)[M]
    t = tpl.image.astype(np.float64)[M]
    w, t = w - w.mean(), t - t.mean()
    den = np.sqrt((w * w).sum() * (t * t).sum())
    return float((w * t).sum() / den) if den > 0 else None


def _print_truth_scores(crop, truth: list[dict], detector, calib) -> None:
    """Pour chaque champion annoté : score du template à la position annotée, anneau lu, détection la plus proche."""
    from vision.detectors.champions import ring_color
    tpls = detector._templates_at_base()
    d = detector.base_diameter
    print(f"\nVérité (annotations.jsonl) : {len(truth)} champion(s) annoté(s)")
    print(f"{'champion':12s} {'annoté x':>8s} {'annoté y':>8s} {'score ici':>9s} {'anneau ici':>10s}   détection de ce champion")
    for a in truth:
        label = a["label"]
        px, py = float(a.get("px", 0)), float(a.get("py", 0))
        tpl = tpls.get(label)
        score = _masked_ncc_at(crop, tpl, px, py) if tpl else None
        ring = ring_color(crop, px, py, d)
        det = next((c for c in detector.last_accepted if c.label == label), None)
        where = (f"à ({det.cx:.0f},{det.cy:.0f}) score {det.score:.2f}, distance {((det.cx - px) ** 2 + (det.cy - py) ** 2) ** 0.5:.0f} px"
                 if det else "aucune")
        print(f"{label:12s} {px:8.1f} {py:8.1f} {score if score is not None else float('nan'):9.2f} {str(ring):>10s}   {where}")


def _dump_frame(out: Path, crop, cands, detector, calib, truth: list[dict] | None = None) -> None:
    """PNG : crop annoté (×2) puis, par champion, le meilleur patch réel (×6) face au template (×6).

    Avec `truth`, une troisième vignette montre le patch à la position annotée (la vérité).
    """
    z = 2
    big = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
    best: dict[str, object] = {}
    for c in cands:
        if c.label not in best or c.score > best[c.label].score:
            best[c.label] = c
    for c in best.values():
        cv2.circle(big, (int(c.cx * z), int(c.cy * z)), int(c.diameter * z / 2), (0, 220, 255), 1)
        cv2.putText(big, f"{c.label} {c.score:.2f}", (int(c.cx * z) - 20, int(c.cy * z) - int(c.diameter * z / 2) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    k = 6
    rows = []
    templates = {t.label: t for b in detector.banks for t in b.templates if b.diameter == detector.base_diameter}
    truth_pos = {a["label"]: (float(a.get("px", 0)), float(a.get("py", 0))) for a in (truth or [])}

    def patch_at(cx: float, cy: float, d: int):
        x0, y0 = int(round(cx - d / 2)), int(round(cy - d / 2))
        x0, y0 = max(0, min(crop.shape[1] - d, x0)), max(0, min(crop.shape[0] - d, y0))
        return crop[y0:y0 + d, x0:x0 + d]

    for label, tpl in templates.items():
        d = tpl.diameter
        c = best.get(label)
        patch = patch_at(c.cx, c.cy, d) if c is not None else np.zeros((d, d, 3), np.uint8)
        tpl_img = tpl.image.copy()
        tpl_img[tpl.mask == 0] //= 3  # zone hors masque assombrie
        tiles = [cv2.resize(patch, (d * k, d * k), interpolation=cv2.INTER_NEAREST),
                 np.full((d * k, 8, 3), 40, np.uint8),
                 cv2.resize(tpl_img, (d * k, d * k), interpolation=cv2.INTER_NEAREST)]
        title = f"{label}: meilleur pic (score {c.score:.2f}, anneau {c.ring})" if c else f"{label}: aucun pic"
        if label in truth_pos:
            tx, ty = truth_pos[label]
            tiles += [np.full((d * k, 8, 3), 40, np.uint8),
                      cv2.resize(patch_at(tx, ty, d), (d * k, d * k), interpolation=cv2.INTER_NEAREST)]
            sc = _masked_ncc_at(crop, tpl, tx, ty)
            title += f"   | template |   annote a ({tx:.0f},{ty:.0f}) score {sc if sc is not None else float('nan'):.2f}"
        else:
            title += "   | template"
        row = np.hstack(tiles)
        band = np.zeros((18, row.shape[1], 3), np.uint8)
        cv2.putText(band, title, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([band, row]))
    w = max([big.shape[1]] + [r.shape[1] for r in rows])
    def pad(img):
        return np.hstack([img, np.zeros((img.shape[0], w - img.shape[1], 3), np.uint8)]) if img.shape[1] < w else img
    montage = np.vstack([pad(big)] + [pad(r) for r in rows])
    cv2.imwrite(str(out), montage)
    print(f"Image écrite : {out}")


def _missing_report(game_dir: Path, manifest, calib, detector, records, args) -> int:
    """Frames où `--missing` est vivant mais non détecté : contexte et planche des patchs à la dernière position connue."""
    label = args.missing
    ids = {c.id for c in manifest.champions}
    if label not in ids:
        # tolérance sur la casse / nom affiché
        for c in manifest.champions:
            if c.id.lower() == label.lower() or c.name.lower() == label.lower():
                label = c.id
    if label not in ids:
        print(f"Champion inconnu : {args.missing} (présents : {', '.join(sorted(ids))})")
        return 1
    d = calib.iconDiameter
    last: tuple[float, float, str] | None = None  # (cx, cy, frameId) de la dernière détection
    misses: list[dict] = []
    alive_frames = detected = 0
    # Une seule échelle (le diamètre calibré) et une frame sur `--step` : assez pour le diagnostic, 6× plus rapide.
    detector.config.scale_probe_frames = 10 ** 9
    base_bank = [b for b in detector.banks if b.diameter == d] or detector.banks[:1]
    detector.banks = base_bank
    detector.base_diameter = base_bank[0].diameter
    sample = records[::max(1, args.step)]
    t_start = time.perf_counter()
    for i, rec in enumerate(sample):
        if i and i % 50 == 0:
            print(f"  … {i}/{len(sample)} frames ({(time.perf_counter() - t_start) / i * 1000:.0f} ms/frame)")
        if label in rec.dead:
            continue
        crop = cv2.imread(str(game_dir / rec.file), cv2.IMREAD_COLOR)
        if crop is None:
            continue
        alive_frames += 1
        detector.detect(crop, calib, rec.t, rec.frameId, exclude=set(rec.dead))
        hit = next((c for c in detector.last_accepted if c.label == label), None)
        if hit is not None:
            detected += 1
            last = (hit.cx, hit.cy, rec.frameId)
            continue
        best = max((c for c in detector.last_candidates if c.label == label), key=lambda c: c.score, default=None)
        near = [c for c in detector.last_accepted if last and ((c.cx - last[0]) ** 2 + (c.cy - last[1]) ** 2) ** 0.5 < 1.2 * d]
        misses.append({"frameId": rec.frameId, "t": rec.t, "last": last, "best": best, "near": near, "crop": crop})
    print(f"{label} : vivant sur {alive_frames} frames, détecté {detected}, manqué {len(misses)}")
    print(f"{'frame':>7s} {'t':>7s} {'dernière pos (frame)':>24s} {'meilleur pic':>22s}  voisins acceptés à moins de 1,2 diamètre de la dernière position")
    for m in misses[:60]:
        b = m["best"]
        lp = f"({m['last'][0]:.0f},{m['last'][1]:.0f}) {m['last'][2]}" if m["last"] else "-"
        bp = f"{b.score:.2f} @({b.cx:.0f},{b.cy:.0f}) {b.ring}" if b else "-"
        nb = ", ".join(f"{c.label}({c.cx:.0f},{c.cy:.0f})" for c in m["near"])
        print(f"{m['frameId']:>7s} {m['t']:7.1f} {lp:>24s} {bp:>22s}  {nb}")
    if len(misses) > 60:
        print(f"… {len(misses) - 60} autres")
    if args.dump and misses:
        k, side = 4, int(2.2 * d)
        tiles = []
        for m in misses:
            if not m["last"]:
                continue
            cx, cy = int(m["last"][0]), int(m["last"][1])
            crop = m["crop"]
            x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
            patch = crop[y0:y0 + side, x0:x0 + side]
            if patch.shape[0] < 4 or patch.shape[1] < 4:
                continue
            tile = cv2.resize(patch, (side * k, side * k), interpolation=cv2.INTER_NEAREST)
            # Dernière position connue (croix) et meilleur pic (cercle).
            cv2.drawMarker(tile, ((cx - x0) * k, (cy - y0) * k), (0, 255, 255), cv2.MARKER_CROSS, 14, 1)
            b = m["best"]
            if b and abs(b.cx - cx) < side and abs(b.cy - cy) < side:
                cv2.circle(tile, (int((b.cx - x0) * k), int((b.cy - y0) * k)), d * k // 2, (255, 255, 255), 1)
            cv2.putText(tile, f"{m['frameId']} {b.score if b else 0:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
            if len(tiles) >= 20:
                break
        if tiles:
            cols = 5
            while len(tiles) % cols:
                tiles.append(np.zeros_like(tiles[0]))
            rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
            cv2.imwrite(str(args.dump), np.vstack(rows))
            print(f"Planche écrite : {args.dump} (croix = dernière position connue, cercle = meilleur pic)")
    return 0


def cmd_debug(args) -> int:
    """Scores détaillés par champion sur une frame (ou statistiques sur tout l'enregistrement)."""
    from collections import defaultdict
    from vision.recorder import load_records
    game_dir = Path(args.game_dir)
    manifest, calib, detector = _detector_for_game(game_dir, args)
    records = load_records(game_dir)
    if not records:
        print("Aucune frame.")
        return 1
    detector.config.scale_probe_frames = 10 ** 9  # pas de verrouillage : on veut voir toutes les échelles
    if args.tune:
        return _tune_icon(game_dir, manifest, calib, records, args)
    if args.missing:
        return _missing_report(game_dir, manifest, calib, detector, records, args)
    if args.frame:
        wanted = [r for r in records if r.frameId == args.frame]
        if not wanted:
            print(f"Frame {args.frame} introuvable.")
            return 1
        rec = wanted[0]
        crop = cv2.imread(str(game_dir / rec.file), cv2.IMREAD_COLOR)
        detector.detect(crop, calib, rec.t, rec.frameId, exclude=set(rec.dead))
        cands = list(detector.last_candidates)
        recovered = [c for c in detector.last_accepted if c.recovered]
        cands += recovered
        accepted = {(c.label, c.cx, c.cy, c.diameter) for c in detector.last_accepted}
        print(f"Frame {rec.frameId} t={rec.t:.1f}s  crop {crop.shape[1]}x{crop.shape[0]}  icône calibrée {calib.iconDiameter} px  "
              f"seuils {detector.config.threshold}/{detector.config.low_threshold}, anneau exigé < {detector.config.ring_required_below}")
        truth = _annotation_for(game_dir, rec.frameId)
        if truth is not None:
            _print_truth_scores(crop, truth, detector, calib)
        if args.dump:
            _dump_frame(Path(args.dump), crop, cands, detector, calib, truth)
        print(f"{'champion':12s} {'score':>6s} {'ajusté':>7s} {'x':>6s} {'y':>6s} {'diam':>4s} {'anneau':>7s} {'attendu':>7s} {'fixe':>4s}  accepté")
        by_label = defaultdict(list)
        for c in cands:
            by_label[c.label].append(c)
        for label in [ch.id for ch in manifest.champions]:
            cs = sorted(by_label.get(label, []), key=lambda c: c.adjusted, reverse=True)[:3]
            if not cs:
                print(f"{label:12s} (aucun pic au-dessus de {detector.config.low_threshold})")
            for c in cs:
                ok = ("OUI (2e passe)" if c.recovered else "OUI") if (c.label, c.cx, c.cy, c.diameter) in accepted else ""
                print(f"{label:12s} {c.score:6.2f} {c.adjusted:7.2f} {c.cx:6.1f} {c.cy:6.1f} {c.diameter:4d} "
                      f"{str(c.ring):>7s} {str(detector.expected_ring(label)):>7s} {'oui' if c.static else '':>4s}  {ok}")
        return 0
    # Statistiques sur tout l'enregistrement : par champion, fréquence de détection, scores, anneaux.
    stats = {ch.id: {"frames": 0, "alive": 0, "scores": [], "best_rejected": [], "rings": defaultdict(int)}
             for ch in manifest.champions}
    step = max(1, len(records) // max(1, args.max_frames))
    used = 0
    for rec in records[::step]:
        crop = cv2.imread(str(game_dir / rec.file), cv2.IMREAD_COLOR)
        if crop is None:
            continue
        used += 1
        dead = set(rec.dead)
        detector.detect(crop, calib, rec.t, rec.frameId, exclude=dead)
        cands, acc = detector.last_candidates, detector.last_accepted
        acc_labels = {c.label for c in acc}
        for c in acc:
            if c.label not in stats:
                continue  # marqueur « ? »
            stats[c.label]["frames"] += 1
            stats[c.label]["scores"].append(c.score)
            stats[c.label]["rings"][str(c.ring)] += 1
            if c.recovered:
                stats[c.label]["recovered"] = stats[c.label].get("recovered", 0) + 1
        for label in stats:
            if label not in dead:
                stats[label]["alive"] += 1
            if label not in acc_labels and label not in dead:
                best = max((c for c in cands if c.label == label), key=lambda c: c.score, default=None)
                if best is not None:
                    stats[label]["best_rejected"].append(best.score)
    print(f"{used} frames analysées (sur {len(records)}), icône calibrée {calib.iconDiameter} px, échelles {[b.diameter for b in detector.banks]}")
    print(f"{'champion':12s} {'équipe':>6s} {'détecté/vivant':>14s} {'2e passe':>8s} {'score moy':>9s} {'score min':>9s} {'anneaux (accepté)':>24s} {'meilleur pic rejeté (moy/max)':>30s}")
    for ch in manifest.champions:
        st = stats[ch.id]
        sc = st["scores"]
        rej = st["best_rejected"]
        rings = ", ".join(f"{k}:{v}" for k, v in sorted(st["rings"].items()))
        print(f"{ch.id:12s} {ch.team:>6s} {st['frames']:7d}/{st['alive']:<6d} {st.get('recovered', 0):8d} "
              f"{(sum(sc) / len(sc)) if sc else 0:9.2f} {min(sc) if sc else 0:9.2f} {rings:>24s} "
              f"{(sum(rej) / len(rej)) if rej else 0:14.2f} / {max(rej) if rej else 0:.2f}")
    print("Lecture : un allié vivant est toujours visible sur la minimap ; détecté sur peu de ses frames « vivant » avec\n"
          "un « meilleur pic rejeté » élevé = seuils trop hauts ou diamètre d'icône faux ; anneau None sur les acceptés =\n"
          "couleur d'anneau non reconnue (voir --ring-mode, --no-ring). Les morts ne sont comptés nulle part\n"
          "(enregistrements antérieurs à la lecture d'isDead : personne n'est marqué mort).")
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
        sp.add_argument("--no-static", action="store_true", help="Ne pas filtrer les pics fixes de la carte")
        sp.add_argument("--no-recover", action="store_true", help="Pas de seconde passe sur les icônes recouvertes")
        sp.add_argument("--no-hidden", action="store_true", help="Ne pas déduire la position des alliés cachés")
        sp.add_argument("--no-ring-only", action="store_true", help="Pas de déduction par l'anneau seul")
        sp.add_argument("--set", action="append", metavar="PARAM=VALEUR",
                        help="Régler un paramètre de DetectorConfig, ex. --set ring_bonus=0.15 (répétable)")
        sp.add_argument("--icon", type=int, help="Forcer le diamètre d'icône (pixels) au lieu de calibration.json")
        sp.add_argument("--icon-zoom", type=float, help="Zoom du portrait avant découpe (défaut : calibration, 1.0)")
        sp.add_argument("--icon-source", choices=["circle", "portrait"], default="circle",
                        help="Templates : icône circulaire du jeu (CommunityDragon, défaut) ou portrait Data Dragon")

    r = sub.add_parser("run", help="capture + détection + enregistrement + WebSocket")
    common_capture(r)
    common_detector(r)
    r.add_argument("--fps", type=float, default=2.0)
    r.add_argument("--crop-margin", type=int, help="Marge autour de la minimap dans le crop traité (défaut : un diamètre d'icône)")
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
    c.add_argument("--set-icon", type=int, metavar="PX",
                   help="Sans fenêtre : écrire ce diamètre d'icône dans calibration.json (toutes les résolutions, ou --resolution)")
    c.add_argument("--resolution", help="Avec --set-icon : résolution à modifier, ex. 2560x1353")
    c.set_defaults(func=cmd_calibrate)

    v = sub.add_parser("view", help="visionneuse d'un enregistrement")
    v.add_argument("game_dir")
    v.add_argument("--zoom", type=int, default=3)
    v.add_argument("--detections", help="Afficher un autre fichier de détections (ex. detections.replay.jsonl)")
    v.add_argument("--replay", action="store_true",
                   help="Rejouer d'abord le détecteur à jour sur l'enregistrement, puis afficher ce résultat")
    v.add_argument("--out", help="Avec --replay : fichier de sortie (défaut detections.replay.jsonl)")
    common_detector(v)
    v.set_defaults(func=cmd_view)

    e = sub.add_parser("eval", help="précision et rappel sur les frames annotées")
    e.add_argument("game_dir")
    e.add_argument("--annotations", help="Fichier d'annotations (défaut : <dossier>/annotations.jsonl)")
    e.add_argument("--tol", type=float, help="Tolérance en pixels (défaut : demi-diamètre d'icône)")
    e.add_argument("--detections", help="Autre fichier de détections (ex. detections.replay.jsonl)")
    e.add_argument("--ignore", help="Champions exclus du calcul, séparés par des virgules (ex. le joueur local)")
    e.add_argument("--errors", nargs="?", const="", metavar="CHAMPION",
                   help="Lister les erreurs frame par frame (toutes, ou celles d'un champion)")
    e.set_defaults(func=cmd_eval)

    rp = sub.add_parser("replay", help="rejouer le détecteur sur un enregistrement (réglage des seuils)")
    rp.add_argument("game_dir")
    common_detector(rp)
    rp.add_argument("--out", help="Nom du fichier de sortie (défaut detections.replay.jsonl)")
    rp.set_defaults(func=cmd_replay)

    dbg = sub.add_parser("debug", help="scores par champion sur une frame, ou statistiques sur l'enregistrement")
    dbg.add_argument("game_dir")
    dbg.add_argument("--frame", help="frameId (ex. 000042) ; sans : statistiques sur tout l'enregistrement")
    dbg.add_argument("--max-frames", type=int, default=100, help="Nombre de frames échantillonnées pour les statistiques")
    common_detector(dbg)
    dbg.add_argument("--tune", action="store_true",
                     help="Chercher le meilleur diamètre d'icône : scores des alliés pour chaque diamètre")
    dbg.add_argument("--zoom", action="store_true", help="Avec --tune : essayer aussi plusieurs zooms de portrait")
    dbg.add_argument("--dump", help="Avec --frame ou --missing : écrire un PNG (patchs réels)")
    dbg.add_argument("--missing", metavar="CHAMPION",
                     help="Parcourir l'enregistrement et montrer où ce champion est manqué (planche avec --dump)")
    dbg.add_argument("--step", type=int, default=2, help="Avec --missing : une frame sur N (défaut 2)")
    dbg.set_defaults(func=cmd_debug)

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
