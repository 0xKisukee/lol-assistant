# Guide d'utilisation du service vision (en clair)

Ce guide explique ce que fait le service et ce que tu vois à chaque commande.
Le README décrit l'installation et les détails techniques ; ici, on parle
usage. Toutes les commandes se tapent dans PowerShell, dans le dossier
`lol-assistant\services\vision`.

## Ce que fait le service, en une minute

Pendant une partie, le service prend une capture d'écran deux fois par
seconde, découpe le carré de la minimap et y reconnaît les dix icônes de
champions, en les comparant aux portraits officiels des champions de la
partie (il connaît la composition grâce à l'API locale du jeu). Pour chaque
icône trouvée il note : quel champion, où sur la carte, avec quelle confiance.
Il diffuse ces détections **en direct**, pendant la partie en cours, sur un
canal local : c'est ce flux que lira la future application quand tu joueras,
pour la partie que tu es en train de jouer. Il enregistre aussi chaque partie
dans un dossier (images de la minimap et détections) : ces enregistrements
servent uniquement à vérifier, mesurer et améliorer le détecteur ; ils ne sont
jamais rejoués dans l'application. Le service ne fait que regarder l'écran :
il ne touche ni au jeu, ni à sa mémoire, ni à tes touches.

Trois mots à connaître :

- **frame** : une image de la minimap prise à un instant donné (deux par seconde) ;
- **détection** : « le détecteur pense que tel champion est là » (un cercle dans la visionneuse) ;
- **annotation** : « voilà où est vraiment tel champion », dit par un humain (une croix jaune). On compare les deux pour mesurer le détecteur.

## Les commandes

### Mettre à jour le code

```powershell
git pull
```

À faire avant chaque séance : récupère les dernières corrections.

### Régler la minimap (une fois par résolution) : `calibrate`

```powershell
uv run python -m vision calibrate --window
```

Le jeu doit être lancé (un Practice Tool suffit) et au premier plan dans les
2 secondes qui suivent la commande. Deux fenêtres s'ouvrent : la capture avec
un **rectangle jaune** (c'est là que le service croit que la minimap est) et
la minimap agrandie avec des **croix** aux endroits où devraient être les
nexus et les tours extérieures. Le but : que le rectangle colle à la minimap
et que les croix tombent sur les icônes de tours.

| Touche | Effet |
|---|---|
| flèches | déplacer le rectangle |
| `+` / `-` | agrandir / réduire le rectangle |
| `o` / `p` | réduire / agrandir le diamètre d'icône (le cercle jaune en haut à gauche doit avoir la taille d'une icône de champion, anneau compris) |
| `t` | recalculer le rectangle en cliquant sur les quatre tours demandées |
| `s` | sauvegarder |
| `q` | quitter |

Sans fenêtre, pour changer seulement le diamètre d'icône :
`uv run python -m vision calibrate --set-icon 33`.

### Enregistrer une partie : `run`

```powershell
uv run python -m vision run --window
```

À lancer dès le chargement de la partie, puis remettre le jeu devant. Le
service affiche :

- « En attente d'une partie… » tant que le jeu n'est pas en partie ;
- la liste des dix champions lus dans l'API, avec `*` sur le tien ;
- une ligne toutes les 5 secondes : numéro de frame, temps de jeu, nombre de
  champions détectés, temps de calcul en millisecondes, noms détectés, morts ;
- « Jeu masqué… capture en pause » si une autre fenêtre passe devant (Alt+Tab) :
  rien n'est enregistré tant que le jeu n'est pas revenu devant.

`Ctrl+C` arrête. L'enregistrement est dans `data\games\<date-heure>` ; le
nom exact est affiché au début (« Enregistrement dans … »). Un enregistrement
contient les petites images de la minimap, les détections et un fichier de
description (champions, résolution, réglages).

### Regarder un enregistrement : `view`

```powershell
uv run python -m vision view data\games\2026-10-20T14-05-34
```

(remplacer par le nom de ton dossier). Une fenêtre montre la minimap
agrandie avec les **cercles** des détections : bleu pour un allié, rouge pour
un ennemi, avec le nom et la confiance (1,00 = certain). Un cercle
**pointillé** marqué « caché? » est un allié que le service n'a pas vu mais
déduit sous l'icône voisine (il est vivant, donc forcément quelque part).
En bas, le bandeau indique la frame courante, le temps de jeu, le nombre de
détections, et les touches.

| Touche | Effet |
|---|---|
| `d` / `a` | frame suivante / précédente |
| `w` / `s` | +10 / −10 frames |
| `p` | lecture automatique (on/off) |
| `n` / `b` | prochaine / précédente frame annotée (le bandeau indique « annotee », ou « annotee VIDE » si elle a été enregistrée sans aucun marqueur) |
| `x` | marquer la frame comme fausse (pour s'en souvenir) |
| `e` | passer en mode annotation (voir ci-dessous) |
| `q` | quitter et sauvegarder |

La toute première frame peut montrer autre chose que le jeu (par exemple le
navigateur) : c'est ce qui était à l'écran à ce moment-là, avance simplement.

### Annoter : dire la vérité pour mesurer le détecteur

Dans la fenêtre de `view`, appuie sur `e`. Des **croix jaunes** apparaissent :
ce sont les annotations, une croix par icône de champion réellement visible.
Comme le détecteur est déjà bon, chaque frame jamais annotée arrive
**préremplie** : ses détections sont copiées en croix. Ton travail est de
corriger les erreurs, rien de plus :

- un cercle est au mauvais endroit ou sur le mauvais champion : **clic droit**
  dessus retire la croix la plus proche ;
- une icône visible n'a pas de cercle : appuie sur le **chiffre** du champion
  (liste tout en bas : `1:Pyke 2:Taric …`, `0` pour le dixième), puis
  **clic gauche** sur l'icône ; la croix se pose là ;
- personne de visible sur la frame : `Suppr` vide les croix ;
- tout est juste : `v` confirme et passe à la frame suivante.

Dès que tu as modifié une frame, elle est conservée automatiquement quand tu
la quittes ; `v` n'est nécessaire que pour confirmer une frame préremplie
que tu n'as pas touchée. Le bandeau indique l'état : « preremplie : v si
tout est juste », « modifiee, conservee », « annotee ». Un clic gauche sans
chiffre appuyé avant ne fait rien (pour pouvoir cliquer sur la fenêtre sans
rien déplacer). `q` écrit tout dans `annotations.jsonl`.

Règle : on annote ce qui est **visible** sur l'image, pas ce qu'on sait. Un
allié entièrement caché sous une autre icône n'a pas de croix.

Combien de frames ? Il n'est pas nécessaire de toutes les faire : 30 à 70
frames par partie, réparties avec `w`, suffisent. Le jalon vise 200 frames
sur trois parties.

### Mesurer : `eval`

```powershell
uv run python -m vision eval data\games\2026-10-20T14-05-34
```

Compare, sur les frames annotées, les cercles aux croix :

- **TP** (vrais positifs) : un cercle avec une croix du même champion au même endroit ;
- **FP** (faux positifs) : un cercle sans croix en face (le détecteur a inventé ou s'est trompé) ;
- **FN** (ratés) : une croix sans cercle (le détecteur n'a pas vu une icône visible) ;
- **précision** = TP / (TP + FP) : quand il dessine un cercle, a-t-il raison ?
- **rappel** = TP / (TP + FN) : voit-il tout ce qui est visible ?

Un champion peut être exclu du calcul : `--ignore Pyke`.

### Comprendre un problème : `debug` et `replay`

```powershell
uv run python -m vision debug data\games\<dossier>
```

Tableau par champion : sur combien de frames il est détecté parmi celles où
il est vivant, ses scores, si la couleur de son anneau est reconnue, et le
meilleur score rejeté quand il n'est pas détecté. Un allié vivant est
toujours sur la minimap ; s'il est rarement détecté, soit il est caché sous
une autre icône (les bots marchent empilés), soit un réglage est faux.

Variantes utiles : `--tune` (cherche le meilleur diamètre d'icône),
`--frame 000100 --dump image.png` (montre les icônes réelles face à leurs
portraits de référence, et le patch à la position annotée), `--missing Nami
--dump nami.png` (où Nami est manquée). `eval … --errors Pyke` liste les
erreurs d'un champion frame par frame. Tous les réglages du détecteur se
changent sans toucher au code avec `--set nom=valeur` (par exemple
`--set threshold=0.5`), et `--no-static`, `--no-recover`, `--no-hidden`
désactivent un mécanisme pour mesurer son apport.

```powershell
uv run python -m vision replay data\games\<dossier>
```

Relance le détecteur sur les images enregistrées, sans rejouer de partie,
par exemple avec d'autres réglages (`--icon 33`, `--threshold 0.5`), et
affiche la latence moyenne. Puis `eval … --detections detections.replay.jsonl`
mesure ce nouveau résultat.

## Où on en est (2026-10-20, soir)

Sur ta partie de référence (78 frames annotées, ton champion compris) :
précision 0,976, rappel 0,969, F1 0,973, environ 190 ms par frame sur ton
PC. Les ennemis visibles sont tous trouvés ou presque, les alliés aussi, et
les alliés cachés sous une autre icône sont déduits et suivis (confiance
réduite, marqués « caché? »). Ce qui reste hors de portée : les cinq
premières secondes à la fontaine (tout le monde empilé) et quelques
empilements très serrés où l'allié caché est placé sur le voisin à 20 px
près. Reste pour clore le jalon 1 : annoter deux autres parties pour
atteindre 200 frames. Ensuite, jalon 2 (détecteur appris sur ces annotations,
capture plus robuste) puis jalon 3 (estimation de la position des ennemis
invisibles, premières aides au joueur).
