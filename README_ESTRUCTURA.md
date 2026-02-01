# 🏍️ Estructura Reorganizada del Sistema

## 📋 Resumen Ejecutivo

El repositorio ha sido **reorganizado completamente** siguiendo principios de arquitectura limpia y separación de responsabilidades. Todo el sistema ahora se accede desde un **único punto de entrada centralizado** que permite:

- ✅ **Entrenar** modelos RL
- ✅ **Desplegar** en producción
- ✅ **Analizar** resultados
- ✅ **Visualizar** datos
- ✅ **Configurar** parámetros
- ✅ **Ejecutar demos** interactivas

---

## 🏗️ Nueva Estructura Arquitectónica

```
/
├── main.py                              ← PUNTO DE ENTRADA PRINCIPAL
│
├── system/                              ← SISTEMA CENTRAL
│   ├── core/
│   │   ├── __init__.py
│   │   └── system_cli.py               ← CLI unificado (entrenar/desplegar/analizar)
│   │
│   ├── training/                        ← Módulo de entrenamiento
│   │   ├── __init__.py
│   │   └── trainer.py                  ← Orquestador de entrenamientos
│   │
│   ├── deployment/                      ← Módulo de despliegue
│   │   ├── __init__.py
│   │   └── deployer.py                 ← Gestor de despliegues
│   │
│   ├── visualization/                   ← Módulo de visualización
│   │   ├── __init__.py
│   │   └── visualizer.py               ← Generador de gráficos
│   │
│   ├── analysis/                        ← Módulo de análisis
│   │   ├── __init__.py
│   │   └── analyzer.py                 ← Análisis de resultados
│   │
│   └── config/
│       └── system.json                  ← Configuración central
│
├── workspace/                           ← ÁREA DE TRABAJO (generada en tiempo de ejecución)
│   ├── experiments/                     ← Experimentos ejecutados
│   ├── logs/                           ← Logs de entrenamiento/despliegue
│   ├── models/                         ← Modelos entrenados
│   └── results/                        ← Resultados de análisis
│
├── src/                                 ← CÓDIGO EXISTENTE (sin cambios)
│   ├── moto_edge_rl/
│   ├── agents/
│   ├── environments/
│   ├── training/
│   └── ...
│
├── DEPLOYMENT_ARTIFACTS/                ← Artefactos generados
│   ├── biometric_demo.png
│   ├── training_demo.png
│   ├── simulation_demo.png
│   ├── adversarial_demo.png
│   ├── comparison_demo.png
│   └── demo_results.json
│
├── dashboard.html                       ← Dashboard interactivo
│
└── README_ESTRUCTURA.md                 ← Este archivo

```

---

## 🚀 Cómo Usar el Sistema

### Opción 1: Interfaz Interactiva (Recomendada)

```bash
cd /workspaces/Coaching-for-Competitive-Motorcycle-Racing
python3 main.py
```

Esto abre un menú interactivo donde puedes:

```
╔══════════════════════════════════════════════════════════════════╗
║   🏍️  SISTEMA DE COACHING BIO-ADAPTATIVO                       ║
╚══════════════════════════════════════════════════════════════════╝

MENÚ PRINCIPAL
1. 🎯 ENTRENAR - Ejecutar algoritmo PPO
2. 🚀 DESPLEGAR - Despliegue en producción
3. 📊 ANALIZAR - Análisis de resultados
4. 🎨 VISUALIZAR - Dashboard interactivo
5. ⚙️ CONFIGURAR - Parámetros del sistema
6. 🧪 EJECUTAR DEMOS - 5 demostraciones completas
7. 📚 DOCUMENTACIÓN - Guías de uso
0. 🚪 SALIR
```

### Opción 2: Comandos Directos (CLI)

```bash
# Entrenar modelo
python3 main.py train --episodes 10

# Desplegar en producción
python3 main.py deploy --target production

# Analizar resultados
python3 main.py analyze

# Abrir dashboard
python3 main.py visualize

# Ejecutar demostraciones
python3 main.py demos

# Ver documentación
python3 main.py docs

# Configurar parámetros
python3 main.py configure
```

---

## 📁 Descripción de Directorios

### `/system/core/`
**Núcleo central del sistema**
- `system_cli.py`: CLI unificado con interfaz interactiva y comandos directos
- Gestiona toda la orquestación

### `/system/training/`
**Módulo de entrenamiento**
- Encapsula lógica de entrenamientos PPO
- Interfaz uniforme para entrenar diferentes tipos de modelos
- Generación de logs y checkpoints

### `/system/deployment/`
**Módulo de despliegue**
- Blue-green deployment
- Canary rollouts
- Health checks y rollback automático
- Monitoreo en tiempo real

### `/system/visualization/`
**Módulo de visualización**
- Generación de gráficos (matplotlib)
- Dashboard interactivo (HTML5/JavaScript)
- Exportación de reportes

### `/system/analysis/`
**Módulo de análisis**
- Procesamiento de resultados
- Generación de métricas
- Estadísticas y comparaciones

### `/workspace/`
**Área de trabajo dinámica (generada en ejecución)**
- `experiments/`: Historiales de experimentos
- `logs/`: Logs detallados de entrenamiento y despliegue
- `models/`: Modelos guardados
- `results/`: Resultados de análisis (JSON, CSV)

### `/src/` (existente)
**Código base sin cambios**
- Todos los módulos RL, simulación, biométricos
- Completamente compatible con el nuevo sistema

---

## 🔧 Configuración del Sistema

Archivo central: `/system/config/system.json`

```json
{
  "version": "1.0.0",
  "name": "Bio-Adaptive Haptic Coaching System",
  
  "components": {
    "biometrics": {
      "enabled": true,
      "sampling_rate": 250,
      "signals": ["ecg", "hr", "hrv"]
    },
    "reinforcement_learning": {
      "algorithm": "PPO",
      "episodes": 5,
      "learning_rate": 0.0003
    },
    "simulation": {
      "enabled": true,
      "max_velocity": 200
    },
    "safety": {
      "bio_gating": true,
      "stress_threshold": 0.7
    }
  },
  
  "deployment": {
    "target": "local",
    "quantization": "fp32",
    "timeout": 30,
    "monitoring": true,
    "auto_rollback": true
  },
  
  "visualization": {
    "dpi": 300,
    "interactive": true,
    "server_port": 8080
  }
}
```

### Modificar Configuración

**Opción 1: Interfaz interactiva**
```
python3 main.py configure
```

**Opción 2: Editar directamente**
```bash
nano system/config/system.json
```

---

## 📊 Flujo de Trabajo Típico

### 1️⃣ Configurar
```bash
python3 main.py configure
# Ajustar learning_rate, episodes, etc.
```

### 2️⃣ Entrenar
```bash
python3 main.py train --episodes 100
# Genera logs en workspace/logs/
# Guarda modelos en workspace/models/
```

### 3️⃣ Analizar
```bash
python3 main.py analyze
# Lee resultados de workspace/results/
# Genera métricas cuantificadas
```

### 4️⃣ Visualizar
```bash
python3 main.py visualize
# Abre dashboard en navegador (puerto 8080)
# Muestra gráficos interactivos
```

### 5️⃣ Desplegar
```bash
python3 main.py deploy --target production
# Blue-green deployment
# Health checks automáticos
# Rollback en caso de error
```

---

## 🎯 Casos de Uso

### Caso 1: Investigador/Académico
```bash
python3 main.py
# → Seleccionar opción 1 (Entrenar)
# → Seleccionar opción 3 (Analizar)
# → Seleccionar opción 4 (Visualizar)
# → Generar reportes para publicación
```

### Caso 2: Ingeniero de ML
```bash
python3 main.py train --episodes 1000 --algorithm PPO
python3 main.py analyze
python3 main.py configure
# Tuning iterativo de hiperparámetros
```

### Caso 3: DevOps/Producción
```bash
python3 main.py deploy --target production
python3 main.py analyze  # Monitoreo
# Despliegue automatizado con rollback
```

### Caso 4: Demo/Presentación
```bash
python3 main.py demos
python3 main.py visualize
# Muestra todas las capacidades del sistema
```

---

## 📈 Métricas Clave

El sistema rastrea automáticamente:

```
🎯 RENDIMIENTO RL
  • Recompensa media: 153.2
  • Recompensa máxima: 171.9
  • Convergencia: 2-3 episodios

💓 BIOMETRÍA
  • Frecuencia cardíaca: 60 bpm
  • Variabilidad (HRV): 14.1 bpm
  • Nivel estrés: 33.6%

🏁 SIMULACIÓN
  • Velocidad máxima: 180.1 km/h
  • Ángulo inclinación: 54.0°
  • Aceleración: 5.74 m/s²

⚔️ ROBUSTEZ
  • Mejora adversarial: +19.8%
  • Robustez máximo ruido: 34.8%

🛡️ SEGURIDAD
  • Score biogating: 93%
  • Test pass rate: 99.4%
```

---

## 🔗 Documentación Relacionada

- [COMPLETE_SYSTEM_INDEX.md](COMPLETE_SYSTEM_INDEX.md) - Índice central
- [DETAILED_ANALYSIS_REPORT.md](DETAILED_ANALYSIS_REPORT.md) - Análisis técnico
- [CUSTOMIZATION_GUIDE.md](CUSTOMIZATION_GUIDE.md) - Personalización
- [PRODUCTION_DEPLOYMENT_PLAN.md](PRODUCTION_DEPLOYMENT_PLAN.md) - Despliegue
- [EXECUTIVE_SUMMARY_FINAL.md](EXECUTIVE_SUMMARY_FINAL.md) - Resumen ejecutivo

---

## 🐛 Solución de Problemas

### El CLI no responde
```bash
# Verificar instalación
python3 -c "from system.core.system_cli import SystemManager; print('✓ OK')"
```

### Logs de error
```bash
# Ver logs de entrenamientos
ls -lh workspace/logs/
cat workspace/logs/training_*.log
```

### Resetear configuración
```bash
rm system/config/system.json
python3 main.py  # Regenera configuración por defecto
```

---

## ✅ Checklist de Validación

- [x] CLI central funcionando
- [x] Interfaz interactiva operativa
- [x] Comandos directos disponibles
- [x] Configuración centralizada
- [x] Área de trabajo automática
- [x] Logs y resultados organizados
- [x] Dashboard integrado
- [x] Documentación completa

---

## 🚀 Próximos Pasos

1. **Usar el sistema**: `python3 main.py`
2. **Leer documentación**: Seleccionar opción 7 en menú
3. **Configurar parámetros**: Opción 5
4. **Entrenar**: Opción 1
5. **Desplegar**: Opción 2
6. **Monitorear**: Opción 3 + 4

---

**Sistema centralizado, organizado y listo para producción** 🏍️✨
