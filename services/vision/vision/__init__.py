"""Service vision de l'assistant League of Legends (lecture seule).

Capture l'écran, calibre la minimap, détecte les champions par template
matching, enregistre les crops et diffuse les détections sur un WebSocket
local. Aucune interaction avec le processus du jeu.
"""

__version__ = "0.1.0"
