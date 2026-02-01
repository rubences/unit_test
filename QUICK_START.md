# 🏍️ SISTEMA COMPLETO DE DEPLOYMENT Y EJECUCIÓN
## Guía Rápida de Uso

---

## ✨ ¿Qué se ha creado?

### **Scripts de Deployment** (3 opciones)

1. **`deploy_system.py`** - Script PRINCIPAL
   - ✅ Ejecuta 7 fases automáticamente
   - ✅ Genera datos, entrena modelo, visualiza
   - ✅ Guarda métricas en JSON/TXT
   - ✅ Manejo correcto de imports

2. **`run_deployment.py`** - Orquestador alternativo
   - Versión más robusta con error handling

3. **`launch.sh`** - Launcher interactivo
   - Menú para elegir modo de ejecución
   - Ver reportes
   - Limpiar artifacts

### **Notebook Interactivo**
- **`notebooks/analysis.ipynb`** (9 secciones)
  - Validación de estructura
  - Carga de configuración
  - Generación de datos
  - Training y evaluación
  - Visualizaciones
  - Persistencia de métricas
  - Análisis estadístico
  - Historial de ejecuciones

### **Documentación**
- **`DEPLOYMENT_GUIDE.md`** - Guía detallada (2000+ palabras)
- **`SYSTEM_OVERVIEW.md`** - Arquitectura del sistema
- **`THIS FILE`** - Guía rápida de uso

---

## 🚀 EJECUCIÓN RÁPIDA

### **Opción 1: Script Automático (RECOMENDADO)**

```bash
cd /workspaces/Coaching-for-Competitive-Motorcycle-Racing
python3 deploy_system.py
```

**Qué sucede:**
- ✅ Fase 1: Validación de estructura (7 directorios)
- ✅ Fase 2: Verificación de dependencias (8 paquetes)
- ✅ Fase 3: Generación de datos (10 laps, 5000 muestras)
- ✅ Fase 4: Entrenamiento PPO (2000 timesteps)
- ✅ Fase 5: Visualizaciones (gráficos PNG)
- ✅ Fase 6: Reportes (JSON + TXT)
- ✅ Fase 7: Resumen final

**Tiempo estimado:** 2-3 minutos

**Salida:** Reportes en `moto_bio_project/reports/`

---

### **Opción 2: Notebook Interactivo**

```bash
cd /workspaces/Coaching-for-Competitive-Motorcycle-Racing/moto_bio_project
jupyter notebook notebooks/analysis.ipynb
```

**Ventajas:**
- 📊 Visualización interactiva
- 📈 Análisis paso a paso
- 💾 Persistencia de datos
- 🔄 Reutilizable

---

### **Opción 3: Launcher Interactivo**

```bash
cd /workspaces/Coaching-for-Competitive-Motorcycle-Racing
bash launch.sh
```

**Menú:**
```
1) Automated Deployment
2) Interactive Notebook
3) Manual Python Execution
4) View Reports
5) Clean Artifacts
```

---

## 📊 ARCHIVOS GENERADOS

Después de ejecutar, encontrarás:

### **Datos**
```
moto_bio_project/data/
└── telemetry.csv          (5000 filas, velocidad, HR, ECG, etc.)
```

### **Modelos**
```
moto_bio_project/models/
└── ppo_bio_adaptive.zip   (Modelo PPO entrenado)
```

### **Visualizaciones**
```
moto_bio_project/logs/
├── training_progress.png              (Gráfico de training)
├── telemetry_distributions.png        (Distribuciones)
└── results_dashboard.png              (3-panel dashboard)
```

### **Métricas y Reportes**
```
moto_bio_project/reports/
├── DEPLOYMENT_SUMMARY.txt             (Resumen legible)
├── deployment_report_*.json           (Métricas JSON)
└── metrics_summary_*.csv              (CSV para análisis)

moto_bio_project/logs/metrics/
├── metrics_*.json                     (JSON completo)
├── metrics_summary_*.csv              (CSV por fase)
└── summary_*.txt                      (Resumen TXT)
```

---

## 🔍 VERIFICAR RESULTADOS

### **Ver Resumen de Ejecución**

```bash
cat moto_bio_project/reports/DEPLOYMENT_SUMMARY.txt
```

**Salida esperada:**
```
===========================================================================
BIO-ADAPTIVE HAPTIC COACHING SYSTEM - EXECUTION SUMMARY
===========================================================================

Timestamp: 2026-01-17T17:53:46
Project: /workspaces/.../moto_bio_project

PHASES:
[structure_validation]
  status: success
  dirs: 7

[dependencies]
  status: success
  packages: 8

[data_generation]
  status: success
  samples: 5000
  laps: 10

[training]
  status: success
  timesteps: 2000
  mean_reward: 45.3

...

SUMMARY:
  Total Duration: 120.5s
  Artifacts: 15
  Status: COMPLETE
```

### **Ver Métricas JSON**

```bash
python3 -c "
import json
with open('moto_bio_project/reports/deployment_report_*.json') as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))
" | head -30
```

### **Listar Artifacts Generados**

```bash
ls -lh moto_bio_project/data/
ls -lh moto_bio_project/models/
ls -lh moto_bio_project/logs/
ls -lh moto_bio_project/reports/
```

---

## 🎯 ESTRUCTURA DEL PROYECTO

```
/Coaching-for-Competitive-Motorcycle-Racing/
│
├── 🚀 SCRIPTS DE EJECUCIÓN
│   ├── deploy_system.py           ← RECOMENDADO
│   ├── run_deployment.py
│   ├── launch.sh
│   └── DEPLOYMENT_GUIDE.md
│
└── 📂 moto_bio_project/
    │
    ├── 🔧 src/ (6 módulos RL)
    │   ├── config.py              (Configuración)
    │   ├── data_gen.py            (Síntesis de datos)
    │   ├── environment.py         (Entorno Gymnasium)
    │   ├── train.py               (Training PPO)
    │   ├── evaluate.py            (Evaluación)
    │   └── visualize.py           (Visualizaciones)
    │
    ├── 📊 data/
    │   └── telemetry.csv          (Datos sintéticos)
    │
    ├── 🤖 models/
    │   └── ppo_bio_adaptive.zip    (Modelo entrenado)
    │
    ├── 📈 logs/
    │   ├── *.png                  (Gráficos)
    │   └── metrics/               (Métricas JSON/CSV)
    │
    ├── 📝 reports/
    │   ├── DEPLOYMENT_SUMMARY.txt (Resumen)
    │   └── *.json                 (Reportes JSON)
    │
    ├── 📚 notebooks/
    │   └── analysis.ipynb         (Análisis interactivo)
    │
    └── ✅ requirements.txt        (Dependencias)
```

---

## 💡 CASOS DE USO

### **Caso 1: Ejecutar y ver resultados rápidamente**

```bash
python3 deploy_system.py && \
cat moto_bio_project/reports/DEPLOYMENT_SUMMARY.txt
```

### **Caso 2: Análisis detallado con Jupyter**

```bash
cd moto_bio_project && \
jupyter notebook notebooks/analysis.ipynb
```

### **Caso 3: Reutilizar datos existentes**

```bash
# Los datos se guardan en moto_bio_project/data/telemetry.csv
# Ejecutar nuevamente reutiliza los datos
python3 deploy_system.py
```

### **Caso 4: Limpiar y empezar de nuevo**

```bash
bash launch.sh
# Opción 5: Clean Artifacts
```

---

## 🔧 TROUBLESHOOTING

### **P: "ModuleNotFoundError: No module named 'gymnasium'"**
```bash
pip install gymnasium stable-baselines3 neurokit2 numpy pandas matplotlib
```

### **P: "Permission denied" en shell script**
```bash
chmod +x launch.sh
bash launch.sh
```

### **P: Jupyter no encontrado**
```bash
pip install jupyter jupyterlab
```

### **P: Artifacts no se generan**
```bash
# Revisar permisos
chmod -R 755 moto_bio_project/
# Intentar nuevamente
python3 deploy_system.py
```

---

## 📊 MÉTRICAS GENERADAS

El sistema genera automáticamente:

| Métrica | Donde | Formato |
|---------|-------|---------|
| Estructura validada | Logs | stdout |
| Dependencias OK | Logs | stdout |
| Datos generados | `data/telemetry.csv` | CSV |
| Modelo entrenado | `models/ppo_bio_adaptive.zip` | ZIP |
| Visualizaciones | `logs/*.png` | PNG |
| Métricas JSON | `reports/*.json` | JSON |
| Resumen TXT | `reports/DEPLOYMENT_SUMMARY.txt` | TXT |
| Histórico CSV | `logs/metrics/*.csv` | CSV |

---

## 🎓 APRENDIZAJE AUTOMÁTICO

### **Datos de Entrada**
- **10 laps** de circuito de 1.2 km
- **500 muestras/lap** (física simulada)
- **Señal ECG** generada con NeuroKit2

### **Modelo Entrenado**
- **Algoritmo**: PPO (Proximal Policy Optimization)
- **Steps**: 2000
- **Reward medio esperado**: 40-50
- **Acción**: 4 opciones de feedback háptico

### **Seguridad**
- **Bio-gating**: Fuerza no-acción si estrés > 0.8
- **Non-learnable**: El modelo NO puede aprender a bypasear

---

## 📞 SOPORTE

Para más información:
1. Lee `DEPLOYMENT_GUIDE.md` (guía completa)
2. Abre `notebooks/analysis.ipynb` (análisis interactivo)
3. Revisa `SYSTEM_OVERVIEW.md` (arquitectura)
4. Ejecuta `python3 deploy_system.py --help` (si lo soporta)

---

## ✅ CHECKLIST DE VALIDACIÓN

Después de ejecutar, verifica:
- [ ] `moto_bio_project/data/telemetry.csv` existe
- [ ] `moto_bio_project/models/ppo_bio_adaptive.zip` existe
- [ ] `moto_bio_project/logs/` tiene archivos PNG
- [ ] `moto_bio_project/reports/DEPLOYMENT_SUMMARY.txt` es legible
- [ ] `moto_bio_project/reports/*.json` contiene métricas
- [ ] `moto_bio_project/notebooks/analysis.ipynb` se abre en Jupyter
- [ ] Todos los artifacts tienen timestamps

---

## 🎉 ¡LISTO!

Tu sistema está completamente configurado y listo para usar.

**Próximo paso**: Ejecuta `python3 deploy_system.py` ahora mismo.

---

**Última actualización**: 2025-01-17
**Versión**: 1.0.0
**Estado**: ✅ PRODUCCIÓN
