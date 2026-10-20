"""Lecture de la Live Client Data API (https://127.0.0.1:2999/liveclientdata/).

Lecture seule. Le certificat du serveur local est signé par l'autorité Riot
« LoL Game Engineering » ; on valide la chaîne avec `riotgames.pem` (embarqué
dans le dépôt, téléchargé depuis
https://static.developer.riotgames.com/docs/lol/riotgames.pem) au lieu de
désactiver la vérification.

Deux usages :
- `LiveClient.wait_for_game()` bloque proprement tant qu'aucune partie n'est
  chargée (le port 2999 ne répond pas hors partie) ;
- `GameClock` suit `gameTime` en arrière-plan (1 Hz) et l'interpole entre
  deux lectures pour horodater chaque capture.
"""

from __future__ import annotations

import logging
import ssl
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

from vision.models import Champion, Team

log = logging.getLogger(__name__)

BASE_URL = "https://127.0.0.1:2999/liveclientdata"
PEM_PATH = Path(__file__).resolve().parent.parent / "riotgames.pem"
RAW_PREFIX = "game_character_displayname_"


class _RiotAdapter(HTTPAdapter):
    """Adaptateur HTTPS qui valide la chaîne avec le certificat Riot.

    `check_hostname=False` n'est utilisé qu'en repli : le certificat feuille
    du jeu peut ne pas porter 127.0.0.1 en SAN, auquel cas Python refuse la
    correspondance d'hôte même si la chaîne est valide. La signature Riot
    reste exigée dans tous les cas.
    """

    def __init__(self, check_hostname: bool = True, **kwargs):
        self._check_hostname = check_hostname
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context(cafile=str(PEM_PATH))
        ctx.check_hostname = self._check_hostname
        ctx.verify_mode = ssl.CERT_REQUIRED
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


@dataclass
class GameStats:
    gameTime: float
    gameMode: str | None = None
    mapName: str | None = None
    mapNumber: int | None = None


@dataclass
class GameInfo:
    """Ce que le service a besoin de savoir d'une partie au démarrage."""

    champions: list[Champion]
    stats: GameStats
    activePlayerName: str | None = None

    @property
    def my_team(self) -> Team | None:
        for c in self.champions:
            if c.isMe:
                return c.team
        return None


class LiveClient:
    """Client HTTP minimal pour la Live Client Data API."""

    def __init__(self, base_url: str = BASE_URL, timeout: float = 1.5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = self._make_session(check_hostname=True)
        self._hostname_fallback_done = False

    @staticmethod
    def _make_session(check_hostname: bool) -> requests.Session:
        s = requests.Session()
        s.mount("https://", _RiotAdapter(check_hostname=check_hostname))
        return s

    def get(self, endpoint: str):
        """GET /liveclientdata/<endpoint>, JSON décodé. Lève `requests.RequestException`."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            r = self._session.get(url, timeout=self.timeout)
        except requests.exceptions.SSLError as e:
            msg = str(e)
            if not self._hostname_fallback_done and ("hostname" in msg.lower() or "ip address mismatch" in msg.lower()):
                log.warning(
                    "Le certificat du jeu ne porte pas 127.0.0.1 : correspondance d'hôte ignorée, "
                    "la signature Riot reste exigée."
                )
                self._hostname_fallback_done = True
                self._session = self._make_session(check_hostname=False)
                r = self._session.get(url, timeout=self.timeout)
            else:
                raise
        r.raise_for_status()
        return r.json()

    def is_game_running(self) -> bool:
        try:
            self.get("gamestats")
            return True
        except requests.RequestException:
            return False

    def gamestats(self) -> GameStats:
        d = self.get("gamestats")
        return GameStats(
            gameTime=float(d.get("gameTime", 0.0)),
            gameMode=d.get("gameMode"),
            mapName=d.get("mapName"),
            mapNumber=d.get("mapNumber"),
        )

    def active_player_name(self) -> str | None:
        try:
            name = self.get("activeplayername")
            return str(name) if name else None
        except requests.RequestException:
            return None

    def dead_players(self) -> set[str]:
        """Noms d'invocateur des joueurs morts (`isDead`) ; un mort n'est pas dessiné sur la minimap."""
        out: set[str] = set()
        for p in self.get("playerlist"):
            if p.get("isDead"):
                out.add(p.get("riotIdGameName") or p.get("summonerName") or "")
        return out

    def playerlist(self, active_name: str | None = None) -> list[Champion]:
        players = self.get("playerlist")
        out: list[Champion] = []
        for p in players:
            raw = p.get("rawChampionName", "")
            cid = raw[len(RAW_PREFIX):] if raw.startswith(RAW_PREFIX) else (p.get("championName") or raw)
            summ = p.get("riotIdGameName") or p.get("summonerName") or ""
            out.append(
                Champion(
                    id=cid,
                    name=p.get("championName") or cid,
                    team=p.get("team", "ORDER"),
                    summonerName=summ,
                    isMe=bool(active_name) and summ == active_name,
                    skinId=int(p.get("skinID", 0) or 0),
                )
            )
        return out

    def game_info(self) -> GameInfo:
        stats = self.gamestats()
        active = self.active_player_name()
        champs = self.playerlist(active)
        if active and not any(c.isMe for c in champs):
            # `activeplayername` peut renvoyer « nom#tag » et `summonerName` l'ancien nom :
            # on retente sur la partie avant le #.
            short = active.split("#")[0]
            for c in champs:
                if c.summonerName.split("#")[0] == short:
                    c.isMe = True
        return GameInfo(champions=champs, stats=stats, activePlayerName=active)

    def wait_for_game(self, poll: float = 1.0, stop: threading.Event | None = None) -> GameInfo | None:
        """Attend qu'une partie soit chargée et que les dix joueurs soient connus."""
        announced = False
        while stop is None or not stop.is_set():
            try:
                info = self.game_info()
                if info.champions:
                    return info
            except requests.RequestException as e:
                if not announced:
                    log.info("En attente d'une partie (le port 2999 répond dès qu'une partie est chargée)… [%s]",
                             type(e).__name__)
                    announced = True
            except ValueError:
                pass  # JSON tronqué pendant le chargement : on réessaie
            time.sleep(poll)
        return None


@dataclass
class GameClock:
    """Suit `gameTime` en arrière-plan et l'interpole entre deux lectures.

    `now()` renvoie le temps de jeu estimé ; `alive` passe à False quand l'API
    ne répond plus (fin de partie), ce qui termine l'enregistrement.
    """

    client: LiveClient
    poll: float = 1.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _game_time: float = 0.0
    _wall: float = 0.0
    _paused: bool = False
    _alive: bool = True
    _failures: int = 0
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    max_failures: int = 5
    _dead: set[str] = field(default_factory=set)
    track_deaths: bool = True

    def start(self) -> "GameClock":
        self._thread = threading.Thread(target=self._loop, name="game-clock", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    @property
    def alive(self) -> bool:
        return self._alive

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                t = self.client.gamestats().gameTime
                dead = self.client.dead_players() if self.track_deaths else set()
                with self._lock:
                    # Si gameTime n'avance plus, la partie est en pause : on gèle l'interpolation.
                    self._paused = (self._wall > 0 and abs(t - self._game_time) < 1e-3)
                    self._game_time, self._wall = t, time.monotonic()
                    self._dead = dead
                    self._failures = 0
                    self._alive = True
            except (requests.RequestException, ValueError):
                with self._lock:
                    self._failures += 1
                    if self._failures >= self.max_failures:
                        self._alive = False
            self._stop.wait(self.poll)

    @property
    def dead(self) -> set[str]:
        """Noms d'invocateur des joueurs morts à la dernière lecture."""
        with self._lock:
            return set(self._dead)

    def now(self) -> float:
        with self._lock:
            if self._wall == 0:
                return 0.0
            if self._paused:
                return self._game_time
            return self._game_time + (time.monotonic() - self._wall)
