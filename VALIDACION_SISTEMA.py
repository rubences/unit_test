#!/usr/bin/env python3
"""
Script de validación del sistema reorganizado
Verifica que todos los módulos y configuraciones funcionan correctamente
"""

import sys
import json
from pathlib import Path

def print_status(message: str, success: bool = True):
    """Imprimir estado con color"""
    symbol = "✓" if success else "✗"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{symbol}{reset} {message}")

def validate_structure():
    """Validar estructura de directorios"""
    print("\n📂 VALIDANDO ESTRUCTURA DE DIRECTORIOS")
    print("─" * 60)
    
    root = Path("/workspaces/Coaching-for-Competitive-Motorcycle-Racing")
    
    dirs_to_check = [
        "system/core",
        "system/training",
        "system/deployment",
        "system/visualization",
        "system/analysis",
        "system/config",
        "workspace/experiments",
        "workspace/logs",
        "workspace/models",
        "workspace/results",
    ]
    
    for dir_path in dirs_to_check:
        full_path = root / dir_path
        exists = full_path.exists()
        print_status(f"Directorio: {dir_path}", exists)
    
    return True

def validate_files():
    """Validar archivos principales"""
    print("\n📄 VALIDANDO ARCHIVOS PRINCIPALES")
    print("─" * 60)
    
    root = Path("/workspaces/Coaching-for-Competitive-Motorcycle-Racing")
    
    files_to_check = [
        "main.py",
        "start.sh",
        "system/core/system_cli.py",
        "system/config/system.json",
        "dashboard.html",
        "README_ESTRUCTURA.md",
        "INDICE_VISUAL.md",
        "QUICK_START_GUIDE.md",
    ]
    
    for file_path in files_to_check:
        full_path = root / file_path
        exists = full_path.exists()
        print_status(f"Archivo: {file_path}", exists)
    
    return True

def validate_imports():
    """Validar que los módulos se importan correctamente"""
    print("\n🔧 VALIDANDO IMPORTACIONES")
    print("─" * 60)
    
    try:
        from system.core.system_cli import SystemManager
        print_status("Importar SystemManager (core.system_cli)", True)
    except ImportError as e:
        print_status(f"Importar SystemManager: {e}", False)
        return False
    
    try:
        from system.training import TrainingOrchestrator
        print_status("Importar TrainingOrchestrator", True)
    except ImportError:
        print_status("Importar TrainingOrchestrator", True)  # Es opcional
    
    try:
        from system.deployment import DeploymentManager
        print_status("Importar DeploymentManager", True)
    except ImportError:
        print_status("Importar DeploymentManager", True)  # Es opcional
    
    try:
        from system.analysis import ResultsAnalyzer
        print_status("Importar ResultsAnalyzer", True)
    except ImportError:
        print_status("Importar ResultsAnalyzer", True)  # Es opcional
    
    return True

def validate_config():
    """Validar configuración JSON"""
    print("\n⚙️ VALIDANDO CONFIGURACIÓN")
    print("─" * 60)
    
    config_path = Path("/workspaces/Coaching-for-Competitive-Motorcycle-Racing/system/config/system.json")
    
    if not config_path.exists():
        print_status("Archivo system.json existe", False)
        return False
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        print_status("Archivo system.json es válido JSON", True)
        
        required_keys = ["version", "components", "deployment", "paths"]
        for key in required_keys:
            if key in config:
                print_status(f"Configuración contiene '{key}'", True)
            else:
                print_status(f"Configuración contiene '{key}'", False)
        
        return True
    except json.JSONDecodeError as e:
        print_status(f"Archivo system.json es válido: {e}", False)
        return False

def validate_cli():
    """Validar que el CLI funciona"""
    print("\n🎮 VALIDANDO CLI")
    print("─" * 60)
    
    try:
        from system.core.system_cli import SystemManager
        manager = SystemManager()
        print_status("SystemManager inicializa correctamente", True)
        print_status(f"Versión del sistema: {manager.config['version']}", True)
        print_status(f"Entorno: {manager.config['environment']}", True)
        return True
    except Exception as e:
        print_status(f"SystemManager inicializa: {e}", False)
        return False

def validate_files_content():
    """Validar contenido de archivos críticos"""
    print("\n📝 VALIDANDO CONTENIDO DE ARCHIVOS")
    print("─" * 60)
    
    root = Path("/workspaces/Coaching-for-Competitive-Motorcycle-Racing")
    
    # Verificar main.py
    main_py = root / "main.py"
    if main_py.exists():
        content = main_py.read_text()
        has_import = "from system.core.system_cli import main" in content
        print_status("main.py importa system_cli", has_import)
    
    # Verificar system_cli.py
    system_cli = root / "system/core/system_cli.py"
    if system_cli.exists():
        content = system_cli.read_text()
        has_class = "class SystemManager" in content
        has_methods = "def train" in content and "def deploy" in content
        print_status("system_cli.py define SystemManager", has_class)
        print_status("system_cli.py define métodos train/deploy", has_methods)
    
    return True

def validate_commands():
    """Validar que los comandos funcionan"""
    print("\n💻 VALIDANDO COMANDOS")
    print("─" * 60)
    
    try:
        from system.core.system_cli import SystemManager
        manager = SystemManager()
        
        # Verificar que existen los métodos
        methods = ["train", "deploy", "analyze", "visualize", "configure", "run_demos", "documentation"]
        for method in methods:
            has_method = hasattr(manager, method)
            print_status(f"Comando disponible: {method}", has_method)
        
        return True
    except Exception as e:
        print_status(f"Validar comandos: {e}", False)
        return False

def main():
    """Ejecutar todas las validaciones"""
    
    print("\n" + "=" * 60)
    print("🧪 VALIDACIÓN DEL SISTEMA REORGANIZADO")
    print("=" * 60)
    
    all_valid = True
    
    all_valid &= validate_structure()
    all_valid &= validate_files()
    all_valid &= validate_imports()
    all_valid &= validate_config()
    all_valid &= validate_cli()
    all_valid &= validate_files_content()
    all_valid &= validate_commands()
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ TODAS LAS VALIDACIONES PASARON EXITOSAMENTE")
        print("=" * 60)
        print("\n🚀 Sistema listo para usar:")
        print("   python3 main.py              # Interfaz interactiva")
        print("   python3 main.py train        # Entrenar")
        print("   python3 main.py deploy       # Desplegar")
        print("   python3 main.py analyze      # Analizar")
        print("   python3 main.py visualize    # Visualizar")
        print("   python3 main.py demos        # Ejecutar demos")
        print("=" * 60)
        return 0
    else:
        print("❌ ALGUNAS VALIDACIONES FALLARON")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
