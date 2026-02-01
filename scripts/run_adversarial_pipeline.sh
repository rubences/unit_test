#!/usr/bin/env bash
# Quick Start: Adversarial Training Pipeline
# ============================================
#
# Este script proporciona comandos para ejecutar el pipeline completo
# de entrenamiento adversario en orden.

set -e

echo "========================================"
echo "ADVERSARIAL TRAINING - QUICK START"
echo "========================================"
echo ""

# Paso 1: Tests
echo "[1/4] Ejecutando tests del SensorNoiseAgent..."
python -m pytest tests/test_adversarial_training.py::TestSensorNoiseAgent -v --tb=short
echo "✓ Tests pasados"
echo ""

# Paso 2: Demo
echo "[2/4] Ejecutando demostración..."
python scripts/adversarial_training_demo.py
echo "✓ Demo completada"
echo ""

# Paso 3: Entrenamiento completo
echo "[3/4] Ejecutando entrenamiento adversario..."
echo "      (Este paso puede tomar 2-4 horas en GPU, 6-12 horas en CPU)"
python -m src.training.adversarial_training
echo "✓ Entrenamiento completado"
echo ""

# Paso 4: Análisis y visualización
echo "[4/4] Generando gráficas y análisis..."
python -m src.analysis.robustness_evaluation
echo "✓ Análisis completado"
echo ""

echo "========================================"
echo "PIPELINE COMPLETE!"
echo "========================================"
echo ""
echo "Resultados guardados en: models/adversarial/"
echo "  - robustness_comparison.png (4 subplots)"
echo "  - robustness_report.txt (análisis detallado)"
echo ""
