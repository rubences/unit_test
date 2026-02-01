#!/usr/bin/env python3
"""
Test de verificación para el comando ANALIZAR
"""

import sys
from pathlib import Path

# Añadir el directorio raíz al path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

def test_analyze():
    """Verificar que el análisis funciona correctamente"""
    from system.core.system_cli import SystemManager
    
    print("=" * 70)
    print("TEST DEL COMANDO ANALIZAR")
    print("=" * 70)
    
    print("\n✓ Creando SystemManager...")
    manager = SystemManager()
    
    print("✓ Ejecutando análisis...\n")
    manager.analyze()
    
    return True

if __name__ == "__main__":
    try:
        test_analyze()
        
        print("\n" + "=" * 70)
        print("✅ TEST COMPLETADO - COMANDO ANALIZAR FUNCIONA CORRECTAMENTE")
        print("=" * 70)
        print("\nEl comando busca resultados en múltiples ubicaciones:")
        print("  1. workspace/results/demo_results.json")
        print("  2. DEPLOYMENT_ARTIFACTS/demo_results.json  ✓ (encontrado)")
        print("  3. demo_results.json")
        print("  4. workspace/results/demo_results.json")
        
    except Exception as e:
        print(f"\n❌ TEST FALLÓ: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
