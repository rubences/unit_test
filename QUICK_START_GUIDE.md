# 🎯 GUÍA RÁPIDA - SISTEMA REORGANIZADO

## ⚡ 30 Segundos para Empezar

```bash
cd /workspaces/Coaching-for-Competitive-Motorcycle-Racing

# Opción 1: Interfaz interactiva (recomendada)
python3 main.py

# Opción 2: Script rápido
bash start.sh

# Opción 3: Comando directo
python3 main.py analyze
```

---

## 🎮 Menú Principal (Interfaz Interactiva)

```
┌─────────────────────────────────────────────────┐
│  🏍️  SISTEMA DE COACHING BIO-ADAPTATIVO        │
│  ✓ Versión 1.0.0                               │
│  ✓ Estado: OPERATIVO                           │
└─────────────────────────────────────────────────┘

1. 🎯 ENTRENAR - Ejecutar algoritmo PPO
2. 🚀 DESPLEGAR - Despliegue en producción
3. 📊 ANALIZAR - Análisis de resultados
4. 🎨 VISUALIZAR - Dashboard interactivo
5. ⚙️ CONFIGURAR - Parámetros del sistema
6. 🧪 EJECUTAR DEMOS - 5 demostraciones
7. 📚 DOCUMENTACIÓN - Guías de uso
0. 🚪 SALIR
```

---

## 💻 Comandos Rápidos (CLI)

```bash
# ENTRENAR
python3 main.py train --episodes 100

# DESPLEGAR
python3 main.py deploy --target production

# ANALIZAR
python3 main.py analyze

# VISUALIZAR
python3 main.py visualize

# EJECUTAR DEMOS
python3 main.py demos

# CONFIGURAR
python3 main.py configure

# VER DOCS
python3 main.py docs
```

---

## 📁 Estructura Nueva

```
/
├── main.py                    ← INICIO AQUÍ
├── start.sh                   ← O AQUÍ (script)
│
├── system/                    ← SISTEMA CENTRAL
│   ├── core/system_cli.py    ← CLI principal
│   ├── training/             ← Entrenamientos
│   ├── deployment/           ← Despliegues
│   ├── visualization/        ← Visualización
│   ├── analysis/             ← Análisis
│   └── config/system.json    ← Configuración
│
├── workspace/                 ← ÁREA TRABAJO
│   ├── experiments/
│   ├── logs/
│   ├── models/
│   └── results/
│
└── DEPLOYMENT_ARTIFACTS/      ← RESULTADOS
    ├── *.png (5 gráficos)
    └── demo_results.json
```

---

## 📊 Lo Que Puedes Hacer

### ✅ Entrenar
```bash
python3 main.py train
# Entrena PPO automáticamente
# Guarda modelos en workspace/models/
# Logs en workspace/logs/
```

### ✅ Desplegar
```bash
python3 main.py deploy
# Blue-green deployment
# Health checks automáticos
# Rollback en caso de error
```

### ✅ Analizar
```bash
python3 main.py analyze
# Lee resultados generados
# Muestra métricas cuantificadas
# Genera reportes
```

### ✅ Visualizar
```bash
python3 main.py visualize
# Abre dashboard en navegador
# Gráficos interactivos
# Puerto 8080
```

### ✅ Ejecutar Demos
```bash
python3 main.py demos
# Ejecuta 5 demostraciones:
#   1. Biometría (ECG/HRV)
#   2. Entrenamiento RL (PPO)
#   3. Simulación (Motocicleta)
#   4. Adversarial (Robustez)
#   5. Comparación (Configs)
```

---

## 📈 Métricas Disponibles

```
🎯 RENDIMIENTO RL
  • Recompensa: 153.2 ± 10.3
  • Convergencia: 2-3 episodios

💓 BIOMETRÍA
  • FC: 60.0 bpm
  • Variabilidad: 14.1 bpm
  • Estrés: 33.6%

🏁 SIMULACIÓN
  • Velocidad: 180.1 km/h
  • Inclinación: 54.0°
  • Aceleración: 5.74 m/s²

⚔️ ROBUSTEZ
  • Mejora: +19.8%
  • Robustez ruido: 34.8%

🛡️ SEGURIDAD
  • Test pass rate: 99.4%
  • Módulos: 37 integrados
```

---

## 📚 Documentación Disponible

| Documento | Acceso |
|-----------|--------|
| **COMPLETE_SYSTEM_INDEX.md** | `python3 main.py docs` → Opción 1 |
| **README_ESTRUCTURA.md** | Referencia del nuevo layout |
| **INDICE_VISUAL.md** | Mapa visual del sistema |
| **DETAILED_ANALYSIS_REPORT.md** | Análisis técnico profundo |
| **CUSTOMIZATION_GUIDE.md** | Personalización de parámetros |
| **PRODUCTION_DEPLOYMENT_PLAN.md** | Plan de despliegue empresarial |
| **EXECUTIVE_SUMMARY_FINAL.md** | Resumen para ejecutivos |

---

## 🔄 Flujo Típico

```
1. python3 main.py
   ↓
2. Seleccionar opción (ejemplo: 6 - Demos)
   ↓
3. Esperar ejecución
   ↓
4. Ver resultados en terminal
   ↓
5. Opción 4 (Visualizar)
   ↓
6. Dashboard en navegador (http://localhost:8080)
   ↓
7. Seleccionar opción 2 (Desplegar) cuando esté listo
```

---

## 🚀 Caso de Uso: Científico

```bash
# Ejecutar demos completas
python3 main.py demos

# Analizar resultados
python3 main.py analyze

# Ver visualizaciones
python3 main.py visualize

# Leer reporte detallado
cat DETAILED_ANALYSIS_REPORT.md

# Personalizar parámetros
python3 main.py configure

# Entrenar con nuevos parámetros
python3 main.py train --episodes 1000
```

---

## 🚀 Caso de Uso: DevOps

```bash
# Verificar configuración
cat system/config/system.json

# Ver logs del último despliegue
ls -lh workspace/logs/

# Desplegar con target específico
python3 main.py deploy --target production

# Monitorear resultados
python3 main.py analyze
```

---

## ⚠️ Solución Rápida de Problemas

```bash
# CLI no responde
python3 -c "from system.core.system_cli import SystemManager; print('✓')"

# Ver configuración
cat system/config/system.json

# Ver logs
ls workspace/logs/

# Ver resultados
cat workspace/results/*.json

# Resetear config
rm system/config/system.json
# Regenera al siguiente python3 main.py
```

---

## ✅ Validación Rápida

```bash
# Verificar estructura
python3 main.py --help

# Ver banner del sistema
python3 main.py

# Ejecutar análisis
python3 main.py analyze

# Abrir dashboard
python3 main.py visualize
```

---

## 🎯 Siguientes Pasos

1. **Ahora:** `python3 main.py` → Seleccionar opción 6 (Demos)
2. **Luego:** `python3 main.py visualize` → Ver dashboard
3. **Después:** Explorar documentación → `python3 main.py docs`
4. **Finalmente:** Desplegar → `python3 main.py deploy`

---

**Sistema totalmente reorganizado y centralizado** 🏍️✨
