"""ARK design tokens — Phase 36 refresh : softer, friendlier, animated.

Évolution :
- Dark mais moins agressif (zinc-inspired CRM)
- Accent rouge ARK gardé comme primary
- Vert électrique signature pour états positifs (#00FF85 inspiré CRM)
- Semantic colors enrichies (success/warning/info distincts)
- Border radii plus arrondis (4-8px)
- Durées d'animation standardisées
- État indicators colorés (idle/training/ok/error/compiled)
"""

# ============================================================
# Couleurs — Fonds zinc-inspired (moins brutalistes que pure black)
# ============================================================

BG_DEEP = "#0E0E11"          # canvas principal (plus doux que pur noir)
BG_PANEL = "#16161D"          # panneaux dock
BG_PANEL_RAISED = "#1C1C25"   # cartes / nodes
BG_OVERLAY = "#22222C"        # menus, popups
BG_HIGHLIGHT = "#2A2A35"      # sélection, hover
BG_INPUT = "#13131A"           # form fields

# Texte : 4 niveaux + 1 inverse
FG_PRIMARY = "#F4F4F5"
FG_SECONDARY = "#C4C4CB"
FG_TERTIARY = "#8E8E98"
FG_DISABLED = "#52525B"
FG_INVERSE = "#0E0E11"  # texte sur fond clair (boutons accent)

# Bordures fines (multi-tier zinc)
BORDER_SUBTLE = "#27272F"
BORDER_DEFAULT = "#3A3A45"
BORDER_STRONG = "#52525B"
BORDER_FOCUS = "#71717A"

# Accent ARK : rouge signature
ACCENT = "#E11D2A"
ACCENT_HOVER = "#F03340"
ACCENT_PRESS = "#B5141F"
ACCENT_DIM = "#3E1018"
ACCENT_SOFT = "rgba(225, 29, 42, 0.12)"  # bg overlay subtil

# Accent secondaire : vert électrique CRM-inspired pour états positifs
ACCENT_GREEN = "#00FF85"
ACCENT_GREEN_DIM = "#04331F"
ACCENT_GREEN_SOFT = "rgba(0, 255, 133, 0.12)"

# États sémantiques distincts (CRM-inspired multi-color)
SUCCESS = "#00FF85"       # vert électrique : OK, compiled, trained
SUCCESS_DIM = "#04331F"
WARN = "#FBBF24"          # amber doux
WARN_DIM = "#3A2A05"
ERROR = "#F87171"         # rouge clair (≠ accent pour erreurs)
ERROR_DIM = "#3F0F0F"
INFO = "#38BDF8"          # cyan doux : training, computing
INFO_DIM = "#0A2E45"

# États nodes (indicateurs status dot)
STATE_IDLE = FG_TERTIARY
STATE_COMPILING = INFO
STATE_TRAINING = INFO
STATE_OK = SUCCESS
STATE_ERROR = ERROR
STATE_WARN = WARN

# Sockets nodes (types de données) — palette élargie
SOCKET_TENSOR = "#A1A1AA"     # zinc-400
SOCKET_NUMBER = "#FBBF24"     # amber
SOCKET_STRING = "#A78BFA"     # violet
SOCKET_IMAGE = "#F472B6"      # pink
SOCKET_MODEL = "#38BDF8"      # cyan
SOCKET_EXEC = ACCENT          # rouge
SOCKET_ANY = "#71717A"


# ============================================================
# Typographie — Arial gardé, mono SF Mono
# ============================================================

FONT_SANS_FAMILY = '"Arial", "Helvetica Neue", "Helvetica", sans-serif'
FONT_MONO_FAMILY = '"Menlo", "JetBrains Mono", "SF Mono", "Consolas", monospace'

# Tailles (px) — légèrement augmentées pour readability
FS_HERO = 20
FS_H1 = 15
FS_H2 = 13
FS_BODY = 12
FS_LABEL = 11
FS_MICRO = 10


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
# Rayons coins — plus arrondis (friendlier)
# ============================================================
R_NONE = 0
R_SMALL = 4    # nodes, buttons standards
R_MEDIUM = 6   # cards, inputs
R_LARGE = 8    # panneaux, modaux
R_PILL = 999   # status dots, badges


# ============================================================
# Largeurs bordures
# ============================================================
B_HAIRLINE = 1
B_REGULAR = 1
B_EMPHASIS = 2
B_FOCUS = 2


# ============================================================
# Animations durations (ms)
# ============================================================
DUR_FAST = 120       # micro interactions (hover, focus)
DUR_NORMAL = 200     # standard transitions
DUR_SLOW = 320       # state changes
DUR_PULSE = 1200     # status pulse breath


# ============================================================
# Easings (QEasingCurve enum names)
# ============================================================
EASE_STANDARD = "OutCubic"   # default fluide
EASE_EMPHASIZED = "OutBack"  # interactions importantes (bounce)
EASE_LINEAR = "Linear"
EASE_IN_OUT = "InOutCubic"


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


# ============================================================
# Shadows (élévations)
# ============================================================
SHADOW_SM = "0 1px 2px rgba(0, 0, 0, 0.4)"
SHADOW_MD = "0 4px 8px rgba(0, 0, 0, 0.5)"
SHADOW_LG = "0 8px 24px rgba(0, 0, 0, 0.6)"
SHADOW_ACCENT = "0 0 0 2px rgba(225, 29, 42, 0.25)"  # focus ring
