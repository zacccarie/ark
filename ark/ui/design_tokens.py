"""ARK design tokens — système graphique unifié.

Philosophie :
- Dark UI uniquement (graphite + noirs profonds).
- Aucun gradient, aucune couleur fluo, aucune saturation gratuite.
- Accent unique : rouge pur (#E11D2A). Pas d'autres couleurs vives.
- Typo Arial famille (sans-serif) pour titres et corps.
- Typo monospace (SF Mono / JetBrains Mono fallback) pour code/valeurs/IDs.
- Espacement multiples de 4px (système 4-8-12-16-24-32-48).
- Coins droits ou très légèrement arrondis (2-4px max). Pas de coins ronds.
- Bordures fines, contrastes mesurés.
"""

# ============================================================
# Couleurs
# ============================================================

# Fonds : graphite profond → graphite clair (5 niveaux pour hiérarchie)
BG_DEEP = "#0A0A0B"          # canvas principal
BG_PANEL = "#111113"          # panneaux dock
BG_PANEL_RAISED = "#16161A"   # cartes / nodes
BG_OVERLAY = "#1C1C21"        # menus, popups
BG_HIGHLIGHT = "#22222A"      # sélection, hover

# Texte : 4 niveaux
FG_PRIMARY = "#F2F2F2"
FG_SECONDARY = "#B6B6BE"
FG_TERTIARY = "#7E7E88"
FG_DISABLED = "#4A4A52"

# Bordures fines
BORDER_SUBTLE = "#22222A"
BORDER_DEFAULT = "#2E2E37"
BORDER_STRONG = "#43434F"

# Accent UNIQUE : rouge pur
ACCENT = "#E11D2A"
ACCENT_HOVER = "#F03340"
ACCENT_PRESS = "#B5141F"
ACCENT_DIM = "#421016"  # background overlay pour zones accentuées

# États sémantiques (mesurés, jamais fluo)
SUCCESS = "#3B8E5C"   # vert sourd
WARN = "#C49A2F"      # jaune-or sourd
ERROR = ACCENT        # rouge = erreur ET accent unique
INFO = "#3E7CB8"      # bleu sourd

# Sockets nodes (types de données)
SOCKET_TENSOR = "#7E7E88"     # gris neutre
SOCKET_NUMBER = "#A6A6AE"
SOCKET_STRING = "#9C8C76"
SOCKET_IMAGE = "#C49A2F"      # jaune sourd
SOCKET_MODEL = "#3E7CB8"      # bleu sourd
SOCKET_EXEC = ACCENT          # flux d'exécution = rouge accent


# ============================================================
# Typographie
# ============================================================

FONT_SANS_FAMILY = '"Arial", "Helvetica Neue", "Helvetica", sans-serif'
FONT_MONO_FAMILY = '"SF Mono", "JetBrains Mono", "Menlo", "Consolas", monospace'

# Tailles (px) — échelle réduite
FS_HERO = 18
FS_H1 = 14
FS_H2 = 12
FS_BODY = 11
FS_LABEL = 10
FS_MICRO = 9


# ============================================================
# Espacement (système 4px)
# ============================================================
S_0 = 0
S_1 = 4
S_2 = 8
S_3 = 12
S_4 = 16
S_5 = 24
S_6 = 32
S_7 = 48
S_8 = 64


# ============================================================
# Rayons coins (droits ou minimal)
# ============================================================
R_NONE = 0
R_SMALL = 2
R_MEDIUM = 4
R_LARGE = 6   # uniquement pour grands panneaux


# ============================================================
# Largeurs bordures
# ============================================================
B_HAIRLINE = 1
B_REGULAR = 1
B_EMPHASIS = 2


# ============================================================
# Z-index sémantiques
# ============================================================
Z_BASE = 0
Z_CONNECTION = 1
Z_NODE = 10
Z_NODE_SELECTED = 11
Z_DRAG = 50
Z_OVERLAY = 100
Z_MODAL = 200
