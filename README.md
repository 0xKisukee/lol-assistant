# lol-assistant

Une seule commande, aucune dépendance : le script interroge la
Live Client Data API que League of Legends expose en local pendant une
partie, et affiche l'état de la partie dans le terminal.

## Lancer

1. Ouvre une partie (une vraie, ou le **Practice Tool** du client, ce qui
   suffit pour tester seul).
2. Dans un terminal, dans le dossier du projet :

   ```
   node v1.mjs
   ```

Tant qu'aucune partie n'est chargée, le script affiche "En attente d'une
partie" et réessaie chaque seconde. Dès que la partie est en cours, il
affiche :

- le mode et le temps de jeu ;
- ton champion, ton niveau, ton or ;
- les dix joueurs avec niveau, K/D/A et CS ;
- deux minuteurs calculés par des règles : prochain dragon, prochain Baron ;
- les derniers événements (kills, tours, objectifs), et ceux qui viennent
  d'arriver.

Arrêter avec `Ctrl+C`.

## Ce que fait le script

- Il appelle `https://127.0.0.1:2999/liveclientdata/allgamedata` une fois
  par seconde. C'est une adresse locale : rien ne sort de ta machine.
- Il ne fait que lire. Il n'envoie aucune commande au jeu et ne touche pas
  à la mémoire du processus, ce qui est la condition pour rester dans les
  règles de Riot.
- Le certificat du serveur local est signé par Riot, pas par une autorité
  connue de Node. Le script désactive la vérification du certificat pour
  cette seule adresse, c'est normal et sans risque puisque c'est ta propre
  machine.

## Voir la réponse brute

Pour comprendre ce que l'API renvoie, ouvre dans un navigateur, pendant une
partie, `https://127.0.0.1:2999/liveclientdata/allgamedata` (accepte
l'avertissement de certificat). Les autres adresses utiles :

- `/liveclientdata/activeplayer` — le joueur local, dont ses stats
  détaillées et ses runes ;
- `/liveclientdata/playerlist` — les dix joueurs ;
- `/liveclientdata/eventdata` — la liste des événements ;
- `/liveclientdata/gamestats` — temps, mode, carte.

## Service vision (jalon 1)

Le service Python de perception (capture d'écran, calibration de la minimap,
détection des dix champions, enregistrement, WebSocket local) vit dans
[`services/vision/`](services/vision/README.md). Son README décrit
l'installation sous Windows et macOS, les commandes (`run`, `calibrate`,
`view`, `eval`) et la procédure pour obtenir les premiers chiffres de
précision et rappel.
