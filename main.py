"""
Punto de entrada principal unificado para todo el sistema
"""

import sys
from pathlib import Path

# Agregar directorios al path
root = Path(__file__).parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

from system.core.system_cli import main

if __name__ == "__main__":
    main()
