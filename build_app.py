"""Build ARK.app via PyInstaller. Run : python build_app.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
ICON_PATH = ROOT / "assets" / "icon.icns"

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--name", "ARK",
    "--windowed",
    "--noconfirm",
    "--clean",
    "--onedir",
    "--osx-bundle-identifier", "fr.zaccarie.ark",
    # Inclure modules dynamiques non détectés
    "--hidden-import", "ark.core.metamorph_bridge",
    "--hidden-import", "ark.core.research_bridge",
    "--hidden-import", "ark.core.diffusion_stubs",
    "--hidden-import", "ark.core.diffusion_real",
    "--hidden-import", "ark.core.plugins",
    "--collect-submodules", "ark",
    # Qt
    "--collect-data", "PySide6",
    # Exclure deps lourdes ML par défaut (utilisateur installe à la demande)
    "--exclude-module", "diffusers",
    "--exclude-module", "transformers",
    "--exclude-module", "huggingface_hub",
    "ark/__main__.py",
]

if ICON_PATH.exists():
    cmd.extend(["--icon", str(ICON_PATH)])

print("[build_app] cmd:", " ".join(cmd))
r = subprocess.run(cmd, cwd=str(ROOT))
sys.exit(r.returncode)
