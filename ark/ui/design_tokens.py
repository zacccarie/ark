"""ARK design tokens — Phase 41 refonte premium CRM-inspired.

Inspiration : CRM rénovation énergétique (PP Neue Machina + Inter, vert
électrique #00FF85, zinc neutrals, R18 cards, letter-spacing négatif).

Philosophie :
- Vert électrique = primary action / state ok (CRM-style)
- Rouge ARK = accent secondaire / erreur
- Zinc multi-tier neutrals
- Border radii généreux (R_MEDIUM 10, R_LARGE 18)
- Letter-spacing négatif titres
- Tabular nums pour chiffres
- Shadows subtiles élévation
- Padding généreux respiration
"""

# ============================================================
# Couleurs — Zinc premium dark
# ============================================================

BG_DEEP = "#09090B"          # canvas principal (zinc-950)
BG_PANEL = "#18181B"          # panneaux dock (zinc-900)
BG_PANEL_RAISED = "#27272A"   # cartes / nodes (zinc-800)
BG_OVERLAY = "#3F3F46"        # menus, popups (zinc-700)
BG_HIGHLIGHT = "#52525B"      # hover (zinc-600)
BG_INPUT = "#0E0E11"           # form fields (très profond)

# Texte : 5 niveaux zinc — contrastes WCAG AA-friendly sur fonds dark
FG_PRIMARY = "#FAFAFA"       # zinc-50
FG_SECONDARY = "#E4E4E7"     # zinc-200 (était zinc-300 trop sombre)
FG_TERTIARY = "#B8B8C0"      # zinc-350 custom (était zinc-400 limite ton-sur-ton)
FG_DISABLED = "#71717A"      # zinc-500 (était zinc-600 invisible)
FG_INVERSE = "#09090B"

# Bordures premium multi-tier
BORDER_SUBTLE = "#27272A"    # zinc-800
BORDER_DEFAULT = "#3F3F46"   # zinc-700
BORDER_STRONG = "#52525B"    # zinc-600
BORDER_FOCUS = "#A1A1AA"     # zinc-400

# Primary = vert électrique signature CRM
PRIMARY = "#00FF85"
PRIMARY_HOVER = "#00E676"
PRIMARY_PRESS = "#00CC6E"
PRIMARY_DIM = "#052E1A"
PRIMARY_SOFT = "rgba(0, 255, 133, 0.08)"

# Action primary = bleu (pour bouton Run/Compile principal style desktop moderne)
ACCENT_BLUE = "#3B82F6"
ACCENT_BLUE_HOVER = "#2563EB"
ACCENT_BLUE_PRESS = "#1D4ED8"
ACCENT_BLUE_DIM = "#0F2240"

# Accent = rouge ARK gardé pour erreurs + accents secondaires
ACCENT = "#E11D2A"
ACCENT_HOVER = "#F03340"
ACCENT_PRESS = "#B5141F"
ACCENT_DIM = "#3E1018"
ACCENT_SOFT = "rgba(225, 29, 42, 0.10)"

# Alias backward-compat (existing code)
ACCENT_GREEN = PRIMARY
ACCENT_GREEN_DIM = PRIMARY_DIM
ACCENT_GREEN_SOFT = PRIMARY_SOFT

# États sémantiques distincts
SUCCESS = PRIMARY              # vert électrique
SUCCESS_DIM = PRIMARY_DIM
WARN = "#FBBF24"               # amber-400
WARN_DIM = "#3A2A05"
ERROR = "#F87171"              # rouge clair (≠ accent)
ERROR_DIM = "#3F0F0F"
INFO = "#38BDF8"               # cyan
INFO_DIM = "#0A2E45"

# États nodes
STATE_IDLE = FG_TERTIARY
STATE_COMPILING = INFO
STATE_TRAINING = INFO
STATE_OK = SUCCESS
STATE_ERROR = ERROR
STATE_WARN = WARN

# Sockets nodes — palette élargie diversifiée
SOCKET_TENSOR = "#A1A1AA"     # zinc-400
SOCKET_NUMBER = "#FBBF24"     # amber
SOCKET_STRING = "#A78BFA"     # violet-400
SOCKET_IMAGE = "#F472B6"      # pink-400
SOCKET_MODEL = "#38BDF8"      # cyan-400
SOCKET_EXEC = PRIMARY         # vert pour flux exécution (CRM)
SOCKET_ANY = "#71717A"


# ============================================================
# Typographie premium
# ============================================================

# Display + sans = Arial (request user Phase 44, garde compat système Mac)
FONT_DISPLAY_FAMILY = '"Arial", "Helvetica Neue", "Helvetica", sans-serif'
FONT_SANS_FAMILY = '"Arial", "Helvetica Neue", "Helvetica", sans-serif'
FONT_MONO_FAMILY = '"Menlo", "JetBrains Mono", "SF Mono", "Consolas", monospace'

# Tailles (px) — premium scale
FS_HERO = 22
FS_H1 = 16
FS_H2 = 14
FS_BODY = 13
FS_LABEL = 12
FS_MICRO = 10


# ============================================================
# Espacement
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
# Rayons coins — CRM-style généreux
# ============================================================
R_NONE = 0
R_SMALL = 6     # buttons standards (était 4)
R_MEDIUM = 10   # inputs / nodes (était 6)
R_LARGE = 18    # cards / panels (CRM exact, était 8)
R_XLARGE = 24
R_PILL = 999


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
DUR_FAST = 150        # micro interactions (hover, focus) — CRM 180ms
DUR_NORMAL = 240      # transitions standard
DUR_SLOW = 360        # state changes
DUR_PULSE = 1400      # status pulse breath


# ============================================================
# Easings (QEasingCurve enum names)
# ============================================================
EASE_STANDARD = "OutCubic"
EASE_EMPHASIZED = "OutBack"
EASE_LINEAR = "Linear"
EASE_IN_OUT = "InOutCubic"
# CRM utilise cubic-bezier(0.22, 1, 0.36, 1) = "easeOutCirc"-like
EASE_CRM = "OutCirc"


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
# Shadows premium (CSS-style strings + Qt QColor RGBA for QGraphicsDropShadow)
# ============================================================
SHADOW_SM = "0 1px 2px rgba(0, 0, 0, 0.4)"
SHADOW_MD = "0 4px 12px rgba(0, 0, 0, 0.5)"
SHADOW_LG = "0 12px 32px rgba(0, 0, 0, 0.6)"
SHADOW_ACCENT = "0 0 0 2px rgba(0, 255, 133, 0.25)"  # focus ring vert

# Pour QGraphicsDropShadowEffect (Qt)
SHADOW_QT_COLOR = (0, 0, 0, 120)  # RGBA tuple
SHADOW_QT_BLUR = 16
SHADOW_QT_OFFSET = (0, 4)
