# lol-vision : service vision de l'assistant (jalon 1)

Service Python autonome qui regarde l'écran pendant une partie de League of
Legends, repère la minimap, y détecte les dix champions par template matching
et diffuse les détections en JSON sur un WebSocket local. Il enregistre aussi
chaque partie (crops de la minimap + détections) pour constituer le jeu de
données du projet.

**Lecture seule.** Le service capture l'écran comme un logiciel de streaming,
lit l'API locale que le jeu expose pendant une partie
(`https://127.0.0.1:2999`), et c'est tout : aucune lecture mémoire, aucune
injection, aucune entrée simulée. Rien ne sort de la machine, à part le
téléchargement des icônes de champions depuis Data Dragon (CDN public de
Riot) et CommunityDragon (miroir public des fichiers du jeu).

## Installation

Le service tourne là où le jeu tourne : **Windows** en pratique (macOS marche
aussi, voir plus bas). Il faut Python 3.12 et [uv](https://docs.astral.sh/uv/).

### Windows (PowerShell)

Pas besoin d'application à installer pour avoir un terminal : PowerShell est
intégré. Menu Démarrer → taper `PowerShell` → Entrée (ou clic droit sur le
bouton Démarrer → « Terminal »). Toutes les commandes ci-dessous se tapent
dans cette fenêtre, une par une.

1. Installer uv (une seule fois), puis **fermer et rouvrir PowerShell** :

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Récupérer le projet. Avec git (`winget install Git.Git` si absent) :

   ```powershell
   git clone <url-du-depot> $HOME\lol-assistant
   cd $HOME\lol-assistant\services\vision
   ```

   Sans git : copier le dossier `services\vision` sur le PC (clé USB, cloud) et
   `cd` dedans.

3. Installer (uv télécharge Python 3.12 tout seul) et vérifier :

   ```powershell
   uv sync --extra dev
   uv run pytest
   ```

4. Régler le jeu en **« Fenêtré sans bordure »** (Options → Vidéo) : la
   capture GDI du plein écran exclusif peut être noire, et les fenêtres de
   `calibrate` / `view` doivent pouvoir s'afficher par-dessus.

Aucune autorisation d'enregistrement d'écran n'est nécessaire sous Windows.
`game.cfg` est lu dans `C:\Riot Games\League of Legends\Config\game.cfg`
(autre lecteur : `--game-cfg D:\...\game.cfg`). `--window` retrouve la
fenêtre « League of Legends (TM) Client » toute seule.

### macOS

```bash
cd services/vision
uv sync --extra dev            # + --extra macos pour --window (pyobjc)
```

`uv` télécharge Python 3.12 s'il manque, crée `.venv/` et installe les
dépendances (opencv-python, numpy, mss, requests, websockets, pillow,
pydantic, pytest). Sans uv : `python3.12 -m venv .venv && . .venv/bin/activate
&& pip install -e '.[dev]'`.

**Autorisation « Enregistrement de l'écran »** : `mss` passe par
CoreGraphics. Sans autorisation, la capture renvoie le fond d'écran sans
aucune fenêtre, **sans erreur**. Il faut autoriser l'application qui lance
Python (Terminal, iTerm, l'app Claude…) : Réglages Système → Confidentialité
et sécurité → Enregistrement de l'écran et audio du système → activer
l'application, puis la relancer. Un bon test : `uv run python -m vision
calibrate` avec n'importe quelle fenêtre ouverte ; si la capture affichée ne
montre que le fond d'écran, l'autorisation manque. Les coordonnées de `mss`
sont en points, les images en pixels physiques (×2 sur Retina) : la
calibration est indexée par la taille de l'image capturée.

### Certificat Riot

Le serveur local du jeu présente un certificat signé par l'autorité Riot
« LoL Game Engineering ». Le fichier `riotgames.pem` (téléchargé depuis
<https://static.developer.riotgames.com/docs/lol/riotgames.pem>) est embarqué
dans le dépôt et sert à **valider** la chaîne, au lieu de désactiver la
vérification. Si le certificat du jeu ne porte pas `127.0.0.1` en SAN, le
client ignore la correspondance d'hôte (un avertissement le dit) mais exige
toujours la signature Riot.

## Commandes

Toutes via `uv run python -m vision <commande>` (ou `uv run vision <commande>`).

| Commande | Rôle |
|---|---|
| `run` | attend une partie, puis capture à 2 i/s, détecte, enregistre et diffuse sur `ws://127.0.0.1:8765` |
| `calibrate` | affiche la capture, laisse ajuster le rectangle de la minimap et le diamètre des icônes, sauvegarde dans `calibration.json` |
| `view <dossier>` | visionneuse image par image d'un enregistrement, marquage des erreurs, annotation |
| `eval <dossier>` | précision et rappel sur les frames annotées |
| `detect <image>` | détection sur une capture PNG (test rapide) |
| `replay <dossier>` | rejoue le détecteur sur les crops enregistrés avec d'autres seuils, écrit `detections.replay.jsonl` |
| `debug <dossier>` | scores par champion sur une frame (`--frame 000042`) ou statistiques sur tout l'enregistrement |
| `assets` | pré-télécharge les portraits Data Dragon |

`--help` sur chaque commande liste les options.

### `run`

```bash
uv run python -m vision run
```

1. Démarre le WebSocket (`--ws-port`, `--no-ws`).
2. Attend qu'une partie soit chargée (« En attente d'une partie… », une
   tentative par seconde). Dès que `/liveclientdata/playerlist` répond, lit
   les dix champions, l'équipe de chacun, le joueur local et `gameMode`.
3. Télécharge (ou lit dans `data/cache/ddragon/<version>/`) les dix portraits
   et les prépare en templates circulaires au diamètre calibré.
4. Capture l'écran à `--fps` (défaut 2), horodate chaque frame avec le
   `gameTime` courant (lu dans l'API à 1 Hz, interpolé entre deux lectures,
   gelé en pause), découpe la minimap, détecte, enregistre, diffuse.
5. Quand l'API ne répond plus (fin de partie), clôt l'enregistrement et
   repasse en attente. `Ctrl+C` arrête.

Options de capture : `--monitor N` (moniteur mss, défaut 1), `--region
x,y,w,h` (points écran), `--window` (fenêtre du jeu via Quartz, macOS). Le
rectangle de la minimap est lu dans `calibration.json` pour la résolution de
la capture ; sinon il est **estimé** depuis `game.cfg` et un avertissement
invite à lancer `calibrate`.

Mode hors-jeu, sans partie : `--source <png|dossier>` remplace l'écran,
`--champions A,B,C,…` remplace l'API (dix noms, les cinq premiers ORDER, puis
CHAOS ; `Nom:CHAOS` force l'équipe, `*Nom` désigne le joueur local),
`--portraits <dossier>` remplace Data Dragon. C'est ce qu'utilise la démo
synthétique plus bas.

### `calibrate`

```bash
uv run python -m vision calibrate            # capture l'écran dans 2 s
uv run python -m vision calibrate --image capture.png
```

Deux fenêtres : la capture avec le rectangle, et la minimap agrandie avec les
**repères attendus** (les deux nexus et les six tours extérieures, en
coordonnées Riot). Touches : flèches ou `i j k l` pour déplacer, `+` `-` pour
la taille, `o` `p` (ou `[` `]`) pour le diamètre d'icône, `f` pas de 1 ou 10 px, `c` colle
au coin, `s` sauvegarde, `q` quitte. Ajuste jusqu'à ce que les croix tombent
sur les icônes de nexus et de tours : c'est la vérification du mapping pixel →
carte. La calibration est enregistrée par résolution de capture
(`"1920x1080"`, `"3456x2234"`…) dans `calibration.json`.

Estimation initiale : `game.cfg` est cherché dans
`C:\Riot Games\League of Legends\Config\game.cfg` (Windows), puis
`/Applications/League of Legends.app/Contents/LoL/Config/game.cfg` et
`~/Applications/…` (macOS) ; `--game-cfg` ou la variable `LOL_GAME_CFG`
donnent un autre chemin. On y lit `MinimapScale` (0..1) et `FlipMiniMap`.
Le côté estimé vaut `hauteur × (0,17 + 0,092 × MinimapScale)` : ce sont des
valeurs communautaires **non vérifiées sur cette machine** (le jeu n'y est pas
installé), d'où l'importance de `calibrate`. Le chemin macOS de `game.cfg`
est lui aussi à confirmer.

### `view`

```bash
uv run python -m vision view data/games/2026-10-20T14-02-11
```

Crop agrandi (`--zoom`, défaut 3), détections dessinées (cercle bleu allié /
rouge ennemi, label, confiance), bandeau avec l'index, le `gameTime` et la
latence. Touches : `d`/`a` ou flèches ±1, `w`/`s` ±10, `p` lecture, `x`
marque ou démarque la frame comme fausse (→ `review.json`), `e` bascule en
annotation, `q` quitte et sauvegarde.

Mode annotation (touche `e`) : une frame jamais annotée est préremplie avec
ses détections ; `1`..`9`,`0` choisissent un champion, clic gauche le place
(ou le déplace), clic droit retire le marqueur le plus proche, `Suppr` vide
la frame (« personne de visible »), `c` recopie les détections si on s'est
trompé. Toute frame **modifiée** est conservée d'office dès qu'on la quitte ;
`v` sert à confirmer une frame préremplie qui était déjà juste, et passe à
la suivante. Tout est écrit dans `annotations.jsonl` à la sortie (`q`).
`w` saute de 10 frames entre deux.

### `eval`

```bash
uv run python -m vision eval data/games/2026-10-20T14-02-11
uv run python -m vision eval <dossier> --annotations autre.jsonl --tol 8
```

Lit `detections.jsonl` et `annotations.jsonl`, apparie par label et distance
(tolérance par défaut : un demi-diamètre d'icône), affiche précision, rappel,
F1 et erreur moyenne de position, globalement et par champion, et écrit
`eval.json`. Les frames non annotées sont ignorées.

## Formats

Un dossier par partie dans `data/games/<horodatage>/` :

- `manifest.json` : `gameId`, `startedAt`, `patch` (version Data Dragon),
  `gameMode`, `mapName`, `resolution`, `calibration`, `champions` (id, nom,
  équipe, `isMe`), `fps`, `detector` (paramètres), `frames`, `endedAt`.
- `frames/000123.webp` : crop de la minimap (qualité 85).
- `detections.jsonl` : une ligne par frame :

  ```json
  {"frameId": "000123", "t": 312.4, "wall": 1760961731.2, "file": "frames/000123.webp",
   "latencyMs": 71.0, "detections": [
     {"t": 312.4, "frameId": "000123", "kind": "champion", "label": "Ahri",
      "pos": {"x": 0.42, "y": 0.61}, "confidence": 0.83, "team": "ORDER",
      "px": {"x": 126.0, "y": 117.0}}]}
  ```

  `pos` est en coordonnées carte normalisées 0..1, **x vers l'est, y vers le
  nord** (orientation des positions de l'API Riot : base ORDER près de (0,0),
  base CHAOS près de (1,1)). `px` est le centre en pixels du crop, pour la
  visionneuse et l'évaluation. Le schéma de `Detection` est celui de
  PLAN.md §5.3 (`kind`, `label`, `pos`, `value`, `confidence`).
- `review.json` : `{"wrong": ["000123", …]}`, frames jugées fausses à l'œil.
- `annotations.jsonl` : vérité terrain, une ligne par frame annotée :

  ```json
  {"frameId": "000123", "champions": [{"label": "Ahri", "px": 126, "py": 117}, {"label": "Garen", "px": 40, "py": 250}]}
  ```

  `px`/`py` en pixels du crop (ce que place la visionneuse) ; `x`/`y`
  normalisés sont aussi acceptés. Liste vide = aucun champion visible.
- `eval.json` : sortie de `eval`.

WebSocket (`ws://127.0.0.1:8765`, un JSON par message, aucun message entrant
n'est traité) : `hello` à la connexion, `game` (champions, calibration, patch)
au début d'une partie, `detections` à chaque frame, `status`
(`waiting`/`running`/`ended`).

## Comment ça marche

- **Calibration** (`calib.py`) : rectangle de la minimap + diamètre d'icône,
  par résolution de capture ; mapping pixel ↔ carte avec repères (nexus,
  tours) pour la vérification visuelle.
- **Templates** (`assets.py`) : l'icône circulaire que le jeu affiche sur la
  minimap (CommunityDragon, miroir public des fichiers du jeu, variante par
  skin via `skinID` de l'API), ou à défaut le portrait Data Dragon (même
  dessin sous un masque rond), ramenée au diamètre d'icône avec un masque
  resserré à 82 % du rayon pour exclure l'anneau d'équipe peint sur le bord.
  `--icon-source portrait` force Data Dragon.
- **Matching** (`matching.py`) : corrélation normalisée masquée (la quantité
  que calcule `cv2.matchTemplate` en `TM_CCOEFF_NORMED` avec masque, mais
  exacte et vingt fois plus rapide) par FFT numpy, les termes qui ne dépendent
  que du masque étant partagés entre templates. Dix templates à trois
  échelles sur un crop de 300 px : ≈ 70 ms sur un Mac M-series, loin sous les
  300 ms visés.
- **Détecteur** (`detectors/champions.py`) : plusieurs pics par champion,
  couleur de l'anneau (bleu allié / rouge ennemi, test par secteurs
  angulaires pour ne pas confondre la rivière avec un anneau, 8 secteurs
  sur 12 suffisent pour tolérer une icône voisine) qui ajuste la confiance
  et est exigée sous un score de 0,85 pour une détection neuve, affectation
  gloutonne (un champion une fois ; deux détections peuvent se chevaucher,
  seule une icône revendiquée deux fois au même endroit est éliminée), suivi
  temporel (dernière position, âge, seuil abaissé près de la dernière
  position, anneau non exigé pour un champion déjà suivi). Les trois
  échelles ne sont essayées que sur les 30 premières frames, puis le
  diamètre qui a donné les meilleurs scores est verrouillé (coût divisé par
  trois).
  Les seuils (`--threshold`, `--low-threshold`, `--scales`, `--ring-mode`,
  `--no-ring`) sont à régler sur des parties réelles.
- **Enregistreur** (`recorder.py`), **serveur** (`server.py`), **visionneuse**
  (`viewer.py`), **évaluation** (`eval.py`), **CLI** (`main.py`).

## Tests

```bash
uv run pytest
```

Les tests utilisent des fixtures synthétiques dans `tests/fixtures/` : dix
« portraits » générés (textures colorées, pas d'assets Riot dans le dépôt),
une minimap avec les dix icônes collées (anneaux bleus et rouges, vérité
terrain dans `minimap_synthetic.json`), le même fond sans icône, et une
capture 1920×1080 factice. Ils couvrent le parsing de `game.cfg`,
l'estimation et le mapping de la calibration (coins, aller-retour, bases),
l'exactitude du moteur FFT contre un calcul direct, le détecteur (dix sur dix,
zéro faux positif sur le fond vide, robustesse à un diamètre faux de ±2 px,
anneaux, suivi, latence), et l'évaluation.
`tests/fixtures/make_fixtures.py` régénère tout (graine fixe).

### Démo de bout en bout sans partie

```bash
uv run python tests/fixtures/make_fixtures.py --frames 40 --out /tmp/seq
cat > /tmp/calib.json <<'JSON'
{"1920x1080": {"width": 1920, "height": 1080, "minimap": {"x": 1620, "y": 780, "w": 300, "h": 300},
 "iconDiameter": 24, "iconZoom": 1.0, "minimapScale": 1.0, "flipped": false, "source": "manual"}}
JSON
uv run python -m vision run --source /tmp/seq --realtime \
  --champions "Aatrox,Bard,Corki,Darius,*Ezreal,Fiora,Garen,Hecarim,Irelia,Jinx" \
  --portraits tests/fixtures/portraits --calibration /tmp/calib.json --games-dir /tmp/games
uv run python -m vision view /tmp/games/<dossier>
uv run python -m vision eval /tmp/games/<dossier> --annotations /tmp/seq/truth.jsonl
```

Sur cette séquence (un champion masqué une frame sur trois), on obtient
précision 1,0 et rappel 1,0.

## Tester avec le jeu

Ce qui n'a pas pu être vérifié sans le jeu, et donc à faire en premier :
le rectangle réel de la minimap, le diamètre des icônes (≈ 8,5 % du côté de
la minimap ?), la couleur des anneaux (bleu = allié et rouge = ennemi, ou
bleu = ORDER ? passer `--ring-mode side` si c'est la seconde), l'icône du
joueur local, le certificat Riot sur `127.0.0.1`, et le chemin de `game.cfg`
sur macOS.

### Practice Tool (seul, immédiat)

1. Client → Jouer → Outil d'entraînement, choisir un champion, lancer.
2. PowerShell : `uv run python -m vision calibrate` puis, dans les 2 s, mettre
   le jeu au premier plan. Ajuster le rectangle sur la minimap, vérifier que
   les croix tombent sur les nexus et les tours, régler le diamètre d'icône
   avec `o` `p` (le gabarit est dessiné en haut à gauche de la fenêtre
   minimap), `s`, `q`.
3. `uv run python -m vision run` : le service doit afficher les champions
   lus dans l'API, puis une ligne toutes les 5 s avec le nombre de champions
   détectés et la latence. En Practice Tool il n'y a qu'un champion (le
   tien) : le test porte sur la capture, la calibration, l'API et le
   certificat, pas encore sur les dix portraits. Ajoute des bots ennemis
   (menu de l'Outil d'entraînement) pour avoir des anneaux rouges.
4. `Ctrl+C`, puis `uv run python -m vision view data/games/<dossier>` pour
   juger à l'œil.

Mode d'affichage : « fenêtré sans bordure » (sous Windows, le plein écran
exclusif peut se capturer en noir) ; en multi-écran, `--monitor 2` ou
`--window`. La capture est une capture d'écran : si une autre fenêtre
recouvre le jeu (Alt+Tab, navigateur), c'est elle qui est capturée. Avec
`--window` sous Windows, le service le détecte (fenêtre du jeu plus au
premier plan) et met la capture en pause, rien n'est enregistré ni diffusé
tant que le jeu n'est pas revenu devant. Sous Windows, `calibrate` capture l'écran 2 s après le lancement :
le temps de cliquer sur la fenêtre du jeu (Alt+Tab).

### Partie personnalisée contre bots (les dix champions)

1. Client → Jouer → Créer une personnalisée, Faille de l'invocateur, ajouter
   9 bots (4 alliés, 5 ennemis), lancer.
2. `uv run python -m vision run` dès le chargement : l'enregistrement couvre
   toute la partie (≈ 100 Mo pour 30 min).
3. Après la partie : `view` pour vérifier que les dix champions sont suivis
   quand ils sont visibles ; `x` sur les frames fausses pour se faire une idée
   rapide.

### Premiers chiffres : précision et rappel sur 200 frames annotées

1. Enregistrer trois parties personnalisées contre bots (critère du jalon 1).
2. Dans `view`, passer en annotation (`e`) et annoter ~70 frames par partie,
   réparties dans le temps (`w` saute de 10 en 10) : la frame est préremplie
   avec les détections, corriger (clic pour déplacer, clic droit pour
   retirer, chiffre puis clic pour ajouter un champion manqué), `v` valide.
   Une frame sans
   champion visible se valide vide. Compter ~1 min par frame au début. On
   annote ce qui est **visible** sur le crop, pas ce qu'on sait : un allié
   entièrement sous une autre icône (les bots marchent empilés en début de
   partie) n'est pas annoté et ne compte donc pas comme un raté ; c'est à
   l'estimation d'état (jalon 3) de le placer.
3. `uv run python -m vision eval data/games/<dossier>` pour chaque partie :
   précision, rappel, F1, erreur moyenne, par champion. `eval.json` garde le
   résultat ; noter la version du détecteur (`manifest.json` → `detector`).
4. Régler ensuite sans relancer de partie. `debug <dossier> --tune` essaie
   une dizaine de diamètres d'icône et donne, pour chacun, le score moyen des
   alliés (toujours visibles sur la minimap) : prendre le meilleur. `debug
   <dossier>` montre, par champion, sur combien de frames il est détecté, ses
   scores, ses anneaux, et le meilleur pic rejeté quand il ne l'est pas (un
   allié rarement détecté avec un pic rejeté élevé = seuils trop hauts ou
   diamètre faux). Puis `replay <dossier> --icon 34 --threshold 0.5` rejoue
   le détecteur sur les crops enregistrés et `eval <dossier> --detections
   detections.replay.jsonl` donne les nouveaux chiffres. Itérer jusqu'au
   meilleur F1, puis reporter les réglages dans `run` (`--icon`, seuils) et
   dans `calibration.json` (via `calibrate`).

## Résultats

Partie personnalisée contre bots du 2026-10-20 (Windows, 2560×1440, minimap
399 px, icônes 33 px), 616 frames à 2 i/s, 78 frames annotées par relecture
des détections préremplies, joueur local (Pyke) placé à la main y compris
sous d'autres icônes. Mesure par `replay` puis `eval` sur ces frames :

| | Valeur |
|---|---|
| Précision | 0,976 (409 vrais, 10 faux positifs) |
| Rappel | 0,969 (13 ratés) |
| F1 | 0,973 |
| Alliés cachés déduits | 18 confirmés par une annotation, 41 non vérifiables (ni vrais ni faux) |
| Joueur local (Pyke, support, souvent sous Tristana) | précision 1,00, rappel 0,88 |
| Ennemis visibles | précision 1,00, rappel 0,95 à 1,00 (Tryndamere 3/4) |
| Latence moyenne par frame (`replay`) | 167 ms sur Mac M-series, ≈ 190 ms sur le PC de test |

Ce qu'il reste : 3 ratés à la fontaine dans les cinq premières secondes
(cinq alliés empilés avant même qu'une piste existe), 6 empilements serrés
où l'allié caché est placé sur l'hôte voisin à 17-33 px de sa position
réelle, et une dizaine de faux positifs dans les frames de fontaine où les
annotations sont incohérentes (icônes empilées détectées à 1,00 mais non
annotées). Les annotations sont versionnées avec le manifest, les détections
et les images de cette partie (`data/games/2026-10-20T14-05-34`, 17 Mo),
qui sert de référence pour tout réglage : `replay` puis `eval`.

Historique sur cette partie : template matching seul 0,80 ; + anneau par
secteurs, pics fixes, traits blancs effacés 0,89 ; + seconde passe sur le
croissant visible et traits fins 0,94 ; + alliés cachés déduits sous l'hôte,
porte de luminosité, pas de continuité sans anneau 0,97 (0,973).

## Limites connues

- La détection repose sur les portraits Data Dragon : skins, icônes
  partiellement recouvertes (pings, autres icônes) et faible échelle de
  minimap baisseront la confiance ; le suivi temporel et l'anneau compensent
  en partie. Le petit détecteur appris est prévu au jalon 2.
- Icônes superposées : le template entier reconnaît l'icône du dessous
  jusqu'à un tiers de recouvrement ; au-delà, une seconde passe cherche les
  champions manquants dans le croissant visible autour de chaque icône
  acceptée (masque du template moins le disque du dessus, déclenchée par des
  pixels d'anneau qui dépassent, anneau de la bonne couleur exigé sur le
  croissant). Mesuré sur fixtures : récupérée jusqu'à 60 % de recouvrement
  (0,4 diamètre d'écart), à 1 px près ; au-delà, invisible pour tout
  détecteur, c'est à l'estimation d'état de la deviner. Cette passe coûte
  40 à 80 ms quand elle se déclenche.
- Morts : l'API (`isDead`) dit qui n'est pas sur la minimap ; ces champions ne
  sont pas cherchés et leur piste est effacée (ils réapparaissent à la base).
  `detections.jsonl` note les morts par frame (`dead`), `debug` compte le
  rappel sur les frames où le champion est vivant.
- Alliés cachés : un allié vivant est toujours dessiné sur la minimap ; s'il
  n'est vu nulle part, il est sous une autre icône. Il est alors placé sur
  l'icône détectée la plus proche de sa dernière position (rayon croissant
  avec l'ancienneté de la piste), avec une confiance de 0,4 et le champ
  `hidden: true` dans la détection (cercle pointillé « caché? » dans la
  visionneuse). Dans les cinq premières secondes, les alliés jamais vus sont
  supposés à leur fontaine. `eval` compte ces déductions à part : confirmées
  si une annotation les recoupe, « non vérifiables » sinon, jamais fausses.
- Luminosité : la corrélation normalisée ignore la luminosité ; une fenêtre
  n'est comparée que si sa moyenne est entre 0,5 et 2 fois celle du template
  (un portrait sombre corrélait à 0,7 avec du terrain noir).
- Bord de la minimap : le crop traité déborde du rectangle calibré d'un
  diamètre d'icône (`cropMargin`, `run --crop-margin`), sinon une icône à la
  fontaine, à cheval sur le bord, ne peut pas être comparée. Les coordonnées
  carte restent celles du rectangle ; les crops enregistrés sont élargis.
- Halo de rappel : pendant un back, un anneau épais et brillant remplace
  l'anneau fin ; il est reconnu comme anneau s'il est vif et quasi complet.
- Ennemis sous une icône : quand un anneau ennemi dépasse d'une icône et qu'un
  cercle s'y ajuste, on prend parmi les ennemis non visibles le portrait le
  plus ressemblant s'il se détache nettement (déduction, `hidden: true`,
  confiance 0,45), sinon un marqueur « ? » de cette équipe (confiance 0,3) :
  « il y a quelqu'un là », sans dire qui.
- Capture par `mss` (CoreGraphics / GDI) : suffisant à 2 i/s ;
  ScreenCaptureKit et DXGI sont prévus plus tard.
- Développé sur macOS sans le jeu : la capture Windows (`mss`, `--window`)
  et le chemin de `game.cfg` sont écrits d'après la documentation, à
  confirmer à la première partie.
