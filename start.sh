#!/bin/bash
# Script de inicio rápido para el sistema

echo "🏍️  Sistema de Coaching Bio-Adaptativo"
echo "════════════════════════════════════════"
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no encontrado"
    exit 1
fi

echo "✓ Python3 disponible"

# Crear directorios si no existen
mkdir -p system/core
mkdir -p system/training
mkdir -p system/deployment
mkdir -p system/visualization
mkdir -p system/analysis
mkdir -p system/config
mkdir -p workspace/{experiments,logs,models,results}

echo "✓ Estructura de directorios creada"

# Instalar dependencias (opcional)
if [ "$1" == "--setup" ]; then
    echo ""
    echo "Instalando dependencias..."
    pip install -q numpy pandas matplotlib seaborn torch gymnasium stable-baselines3
    echo "✓ Dependencias instaladas"
fi

# Iniciar CLI
echo ""
echo "Iniciando Sistema..."
echo "════════════════════════════════════════"
echo ""

python3 main.py
