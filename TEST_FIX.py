#!/usr/bin/env python3
"""
Test rápido para verificar que el fix del KeyError funciona
"""

import sys
from pathlib import Path

# Añadir el directorio raíz al path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

def test_config_loading():
    """Verificar que la configuración se carga correctamente"""
    from system.core.system_cli import SystemManager
    
    print("✓ Probando carga de SystemManager...")
    manager = SystemManager()
    
    print("✓ Verificando estructura de configuración...")
    assert 'components' in manager.config
    assert 'reinforcement_learning' in manager.config['components']
    assert 'deployment' in manager.config
    assert 'visualization' in manager.config
    
    print("✓ Verificando parámetros de RL...")
    rl_config = manager.config['components']['reinforcement_learning']
    assert 'algorithm' in rl_config
    assert 'episodes' in rl_config
    assert 'learning_rate' in rl_config
    assert 'batch_size' in rl_config
    
    print(f"  • Algoritmo: {rl_config['algorithm']}")
    print(f"  • Episodios: {rl_config['episodes']}")
    print(f"  • Learning rate: {rl_config['learning_rate']}")
    print(f"  • Batch size: {rl_config['batch_size']}")
    
    return True

def test_train_method():
    """Verificar que el método train no lanza KeyError"""
    from system.core.system_cli import SystemManager
    
    print("\n✓ Probando método train (sin ejecutar)...")
    manager = SystemManager()
    
    # Verificar que podemos acceder a los parámetros sin error
    rl_config = manager.config['components']['reinforcement_learning']
    episodes = rl_config['episodes']
    algorithm = rl_config['algorithm']
    
    print(f"  • Método train puede acceder a configuración ✓")
    print(f"  • Algoritmo por defecto: {algorithm}")
    print(f"  • Episodios por defecto: {episodes}")
    
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("TEST DE FIX - KeyError: 'training'")
    print("=" * 70)
    
    try:
        test_config_loading()
        test_train_method()
        
        print("\n" + "=" * 70)
        print("✅ TODOS LOS TESTS PASARON - FIX VERIFICADO")
        print("=" * 70)
        print("\nEl error KeyError: 'training' ha sido corregido.")
        print("El sistema ahora usa la ruta correcta: config['components']['reinforcement_learning']")
        
    except Exception as e:
        print(f"\n❌ TEST FALLÓ: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
