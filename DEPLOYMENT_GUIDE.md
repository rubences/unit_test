# 🏍️ Sistema de Coaching Adaptativo Háptico para Carreras Competitivas
## Deployment y Ejecución Completa

### 📋 Tabla de Contenidos
1. [Descripción General](#descripción-general)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Instalación y Setup](#instalación-y-setup)
4. [Ejecución del Sistema](#ejecución-del-sistema)
5. [Archivos Generados](#archivos-generados)
6. [Análisis de Resultados](#análisis-de-resultados)

---

## 📖 Descripción General

Este proyecto implementa un **sistema de coaching adaptativo con retroalimentación háptica** para conducción de motocicletas en competencias. Utiliza:

- **Aprendizaje por Refuerzo**: Algoritmo PPO (Proximal Policy Optimization)
- **Síntesis de Datos**: Telemetría realista + señales ECG de NeuroKit2
- **Mecanismo de Seguridad Bio**: Non-learnable bio-gating basado en estrés fisiológico
- **Visualización**: Dashboard de 3 paneles listo para publicación

### 🎯 Características Principales

✅ **Generación de Datos Sintéticos**: 10 laps con 500 muestras/lap (5000 total)
✅ **Entrenamiento Automático**: PPO con callbacks y checkpoints
✅ **Evaluación Robusta**: Múltiples episodios con métricas completas
✅ **Visualizaciones**: Publicación-ready (300 DPI)
✅ **Métricas Persistentes**: JSON, CSV, TXT con timestamps
✅ **Orquestación Completa**: Script maestro que ejecuta todo

---

## 📁 Estructura del Proyecto

```
/Coaching-for-Competitive-Motorcycle-Racing/
│
├── 🚀 run_deployment.py              # Script maestro de deployment
│
├── 📂 moto_bio_project/
│   ├── src/                          # 6 módulos de RL
│   │   ├── config.py                # Configuración centralizada
│   │   ├── data_gen.py               # Síntesis de telemetría + ECG
│   │   ├── environment.py            # Entorno Gymnasium con bio-gating
│   │   ├── train.py                  # Entrenamiento PPO
│   │   ├── evaluate.py               # Evaluación de modelo
│   │   └── visualize.py              # Visualizaciones
│   │
│   ├── models/                       # Artifacts guardados
│   │   └── ppo_bio_adaptive.zip      # Modelo entrenado
│   │
│   ├── data/                         # Datos sintéticos
│   │   ├── telemetry.csv             # Trayectoria de moto
│   │   └── ecg_signal.npy            # Señal ECG de 500 Hz
│   │
│   ├── logs/                         # Métricas y visualizaciones
│   │   ├── metrics/                  # JSON/CSV/TXT
│   │   ├── artifacts/                # PNG/ZIP
│   │   └── notebook_run.log          # Log de ejecución
│   │
│   ├── notebooks/                    # Análisis interactivo
│   │   └── analysis.ipynb            # 9 secciones de análisis
│   │
│   ├── scripts/                      # Scripts auxiliares
│   │   ├── deploy_complete.py        # Orquestador alternativo
│   │   └── ...otros...
│   │
│   ├── reports/                      # Reportes finales
│   │   ├── deployment_report_*.json  # Reporte JSON
│   │   └── DEPLOYMENT_SUMMARY.txt    # Resumen TXT
│   │
│   └── requirements.txt              # Dependencias
│
└── 📚 docs/                          # Documentación
    └── IMPLEMENTATION_GUIDE.md
```

---

## ⚙️ Instalación y Setup

### 1️⃣ Clonar/Descargar Proyecto

```bash
cd /workspaces/Coaching-for-Competitive-Motorcycle-Racing
```

### 2️⃣ Instalar Dependencias

```bash
# Opción A: Desde archivo requirements.txt
pip install -r moto_bio_project/requirements.txt

# Opción B: Instalación manual
pip install numpy pandas matplotlib gymnasium stable-baselines3 neurokit2 scipy scikit-learn
```

### 3️⃣ Verificar Estructura

```bash
python -c "
from pathlib import Path
p = Path('moto_bio_project')
print('✅ Estructura OK' if p.exists() else '❌ Estructura incompleta')
"
```

---

## 🚀 Ejecución del Sistema

### Opción 1: Ejecutar Script Maestro (RECOMENDADO)

```bash
# Desde la raíz del workspace
python run_deployment.py
```

**Qué sucede**:
- ✅ Fase 1: Validar estructura
- ✅ Fase 2: Verificar dependencias
- ✅ Fase 3: Generar datos (10 laps)
- ✅ Fase 4: Entrenar PPO (3000 timesteps)
- ✅ Fase 5: Generar visualizaciones
- ✅ Fase 6: Crear reportes JSON/TXT
- ✅ Fase 7: Resumen final

**Tiempo estimado**: 2-5 minutos

**Salida esperada**:
```
================================================================================
🏍️  BIO-ADAPTIVE HAPTIC COACHING SYSTEM - DEPLOYMENT ORCHESTRATOR
================================================================================
Fecha/Hora: 2025-01-17 10:30:45
Raíz: /workspaces/Coaching-for-Competitive-Motorcycle-Racing/moto_bio_project

================================================================================
FASE 1: VALIDACIÓN DE ESTRUCTURA
================================================================================
✅ src: /workspaces/Coaching-for-Competitive-Motorcycle-Racing/moto_bio_project/src
✅ models: ...
✅ Modelo entrenado y guardado
  • Mean Reward: 45.23

================================================================================
FASE 7: RESUMEN FINAL
================================================================================

📊 RESULTADO FINAL:
  • Fases completadas: 7/7
  • Tiempo total: 234.56s
  • Raíz del proyecto: /workspaces/Coaching-for-Competitive-Motorcycle-Racing/moto_bio_project

📁 ARTIFACTS GENERADOS:
  • Modelos: /workspaces/Coaching-for-Competitive-Motorcycle-Racing/moto_bio_project/models
  • Datos: /workspaces/Coaching-for-Competitive-Motorcycle-Racing/moto_bio_project/data
  • Logs: /workspaces/Coaching-for-Competitive-Motorcycle-Racing/moto_bio_project/logs
  • Reportes: /workspaces/Coaching-for-Competitive-Motorcycle-Racing/moto_bio_project/reports

================================================================================
✅ DEPLOYMENT COMPLETADO EXITOSAMENTE
================================================================================
```

### Opción 2: Ejecución Interactiva (Jupyter Notebook)

```bash
cd moto_bio_project
jupyter notebook notebooks/analysis.ipynb
```

**Secciones disponibles**:
1. Validación de estructura
2. Carga de configuración
3. Generación de datos (10 laps)
4. Setup del entorno
5. Entrenamiento PPO (5000 steps)
6. Evaluación (3 episodios)
7. Persistencia de métricas
8. Estadísticas de telemetría
9. Historial de ejecuciones

---

## 📊 Archivos Generados

### Después de ejecutar `python run_deployment.py`:

#### 1. **Datos** (`moto_bio_project/data/`)
```
telemetry.csv           # Trayectoria de motocicleta (5000 filas)
  - speed_kmh, lean_angle_deg, g_force, heart_rate, ecg, fatigue, stress, etc.
```

#### 2. **Modelos** (`moto_bio_project/models/`)
```
ppo_bio_adaptive.zip    # Modelo PPO entrenado (RL policy)
```

#### 3. **Logs y Métricas** (`moto_bio_project/logs/`)
```
metrics/
  ├── metrics_20250117_103045.json     # Métricas completas JSON
  ├── metrics_summary_20250117_103045.csv  # Resumen CSV
  └── summary_20250117_103045.txt      # Resumen legible

artifacts/
  ├── training_progress.png            # Gráfico de training
  ├── telemetry_distributions.png      # Distribuciones
  └── results_dashboard.png            # Dashboard 3-panel
```

#### 4. **Reportes** (`moto_bio_project/reports/`)
```
deployment_report_20250117_103045.json  # Reporte JSON completo
DEPLOYMENT_SUMMARY.txt                   # Resumen ejecutivo
```

#### 5. **Notebook** (`moto_bio_project/notebooks/`)
```
analysis.ipynb  # Análisis interactivo con 9 secciones
```

---

## 📈 Análisis de Resultados

### 1️⃣ Revisar Métricas

```bash
# Ver resumen TXT
cat moto_bio_project/reports/DEPLOYMENT_SUMMARY.txt

# Ver JSON (para programación)
python -c "
import json
with open('moto_bio_project/reports/DEPLOYMENT_SUMMARY.txt', 'r') as f:
    print(f.read())
"
```

### 2️⃣ Análisis en Python

```python
import json
import pandas as pd

# Cargar último reporte
with open('moto_bio_project/reports/DEPLOYMENT_SUMMARY.txt') as f:
    summary = json.load(f)

# Metrics en CSV
df = pd.read_csv('moto_bio_project/logs/metrics/metrics_summary_*.csv')
print(df)
```

### 3️⃣ Ejecutar Notebook Interactivo

```bash
cd moto_bio_project
jupyter notebook notebooks/analysis.ipynb

# Luego ejecutar celdas:
# 1. Run all para ejecución completa
# 2. Visualizar gráficos
# 3. Ver historial de ejecuciones
```

---

## 🎯 Flujo Completo de Ejecución

```
┌─────────────────────────────────────────────────┐
│  python run_deployment.py                       │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
    ✅ Fase 1          ✅ Fase 2
   Estructura        Dependencias
        │                     │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
    ✅ Fase 3          ✅ Fase 4
   Datos Gen.          Training
    (10 laps)         (3000 steps)
        │                     │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
    ✅ Fase 5          ✅ Fase 6
   Visualización      Reportes
                      JSON+TXT
        │                     │
        └──────────┬──────────┘
                   │
            ✅ Fase 7
          Resumen Final
                   │
        ┌──────────v──────────┐
        │ 📊 ARTIFACTS SAVED   │
        ├──────────────────────┤
        │ • models/*.zip       │
        │ • logs/metrics/*.json│
        │ • logs/*.png         │
        │ • reports/*.txt      │
        └──────────────────────┘
```

---

## 🔧 Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'gymnasium'"

```bash
pip install gymnasium stable-baselines3 neurokit2
```

### Problema: "CUDA not available"

No es necesario GPU. El sistema corre en CPU perfectamente.

### Problema: "Permission denied run_deployment.py"

```bash
chmod +x run_deployment.py
python run_deployment.py  # Ejecutar con python
```

### Problema: Espacio insuficiente en disco

Limpieza de logs viejos:
```bash
rm -rf moto_bio_project/logs/*
rm -rf moto_bio_project/data/*
```

---

## 📞 Próximos Pasos

1. ✅ **Ejecutar deployment**: `python run_deployment.py`
2. 📊 **Revisar métricas**: `cat moto_bio_project/reports/DEPLOYMENT_SUMMARY.txt`
3. 📈 **Análisis detallado**: Abrir `notebooks/analysis.ipynb` en Jupyter
4. 🧪 **Validación**: Verificar que todos los artifacts existan
5. 🚀 **Producción**: Integrar modelo con haptic feedback hardware

---

## 📚 Referencias

- **RL Framework**: [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- **Gymnasium**: [OpenAI Gymnasium](https://gymnasium.farama.org/)
- **NeuroKit2**: [NeuroKit2 ECG](https://neurokit2.readthedocs.io/)
- **Paper**: Bio-Cybernetic Adaptive Haptic Coaching System

---

**Última actualización**: 2025-01-17
**Versión**: 1.0.0
**Estado**: ✅ Producción
