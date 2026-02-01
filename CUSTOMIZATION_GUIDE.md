# ⚙️ CONFIGURACIONES PERSONALIZABLES - Sistema Coaching Adaptativo

**Última actualización:** 17 Enero 2026  
**Versión:** 2.0.0 - CUSTOMIZABLE

---

## 🎛️ TABLA DE CONTROL - PARÁMETROS AJUSTABLES

### Sección 1: PARÁMETROS DE ENTRENAMIENTO RL

```yaml
# archivo: moto_bio_project/src/config.py

TRAINING:
  # Algoritmo
  algorithm: "PPO"              # Opciones: PPO, A2C, TRPO, SAC
  learning_rate: 3e-4           # Rango: [1e-5, 1e-2]
  batch_size: 64                # Rango: [16, 256]
  n_steps: 2048                 # Rango: [128, 8192]
  
  # Episodios
  total_timesteps: 1_500_000    # Rango: [100k, 10M]
  n_epochs: 10                  # Rango: [1, 20]
  clip_range: 0.2               # Rango: [0.1, 0.5]
  
  # Regularización
  ent_coef: 0.01                # Rango: [0.0, 0.1]
  vf_coef: 0.5                  # Rango: [0.1, 1.0]
  gamma: 0.99                   # Rango: [0.9, 0.999]
  gae_lambda: 0.95              # Rango: [0.9, 0.99]
```

**Recomendaciones:**
- **Aprendizaje rápido:** `learning_rate=5e-4`, `n_steps=1024`
- **Aprendizaje estable:** `learning_rate=3e-4`, `n_steps=2048`
- **Producción robusta:** `learning_rate=1e-4`, `n_steps=4096`, `clip_range=0.1`

---

### Sección 2: PARÁMETROS BIOMÉTRICOS

```yaml
# archivo: moto_bio_project/src/config.py

BIOMETRICS:
  # ECG
  ecg:
    sampling_rate: 250            # Hz: [100, 1000]
    noise_level: 0.1              # Amplitud: [0.0, 0.5]
    heart_rate_base: 70           # bpm: [40, 200]
    heart_rate_variation: 20      # bpm: [0, 50]
  
  # HRV
  hrv:
    rmssd_threshold_low: 10       # ms: [5, 50]
    rmssd_threshold_high: 100     # ms: [50, 200]
    stress_scale_factor: 1.0      # Factor: [0.5, 2.0]
  
  # Análisis
  analysis:
    window_size: 60               # segundos: [30, 300]
    overlap: 0.5                  # Fracción: [0.0, 0.9]
    moving_average_window: 10     # samples: [1, 100]
```

**Perfiles Preconfigurados:**
```
CASUAL (Principiante):
  sampling_rate: 100
  noise_level: 0.2
  stress_scale_factor: 1.5

COMPETITIVE (Intermedio):
  sampling_rate: 250
  noise_level: 0.1
  stress_scale_factor: 1.0

PROFESSIONAL (Experto):
  sampling_rate: 500
  noise_level: 0.05
  stress_scale_factor: 0.8
```

---

### Sección 3: PARÁMETROS DE SEGURIDAD (Bio-gating)

```yaml
# archivo: src/safety/bio_gating.py

BIO_GATING:
  # Umbrales de activación
  stress_threshold: 0.7           # Fracción: [0.5, 0.9]
  heart_rate_max: 180             # bpm: [150, 220]
  heart_rate_min: 40              # bpm: [30, 60]
  
  # Restricciones de acción
  max_throttle_stressed: 0.7      # Fracción: [0.3, 1.0]
  max_lean_stressed: 0.8          # Fracción: [0.5, 1.0]
  max_brake_stressed: 0.9         # Fracción: [0.5, 1.0]
  
  # Modo de activación
  activation_mode: "soft"         # Opciones: hard, soft, adaptive
  soft_range: 0.1                 # Transición suave: [0.0, 0.3]
  
  # Recuperación
  recovery_time: 5                # segundos: [1, 30]
  recovery_method: "gradual"      # Opciones: gradual, step, exponential
```

**Estrategias de Bio-gating:**

| Estrategia | Seguridad | Rendimiento | Uso |
|-----------|-----------|-------------|-----|
| **Hard** | Máxima | Bajo | Principiantes |
| **Soft** | Alta | Medio | Competición |
| **Adaptive** | Muy Alta | Muy Alto | Profesional |

---

### Sección 4: PARÁMETROS DE SIMULACIÓN

```yaml
# archivo: simulation/motorcycle_env.py

MOTORCYCLE:
  # Física
  mass: 200                       # kg: [150, 300]
  wheelbase: 1.4                  # metros: [1.0, 1.8]
  cg_height: 0.6                  # metros: [0.5, 0.8]
  
  # Motor
  max_speed: 300                  # km/h: [200, 350]
  max_torque: 120                 # Nm: [50, 200]
  acceleration_factor: 1.0        # Escala: [0.5, 2.0]
  
  # Dinámica lateral
  max_lean: 65                    # grados: [45, 80]
  lean_speed: 2.0                 # °/s: [1.0, 5.0]
  
  # Fricción
  friction_coefficient: 1.2       # µ: [0.8, 1.5]
  road_grip: 1.0                  # Factor: [0.5, 1.5]

ENVIRONMENT:
  # Condiciones
  wind_speed: 0                   # km/h: [0, 40]
  rain_factor: 0                  # Fracción: [0, 1.0]
  temperature: 25                 # °C: [0, 50]
  
  # Pista
  track_type: "circuit"           # Opciones: circuit, highway, offroad
  grip_variation: 0.05            # Aleatoriedad: [0, 0.2]
```

**Presets de Simulación:**
```
PRACTICE (Entrenamiento):
  max_lean: 60°
  friction_coefficient: 1.3
  wind_speed: 0 km/h

QUALIFYING (Calificación):
  max_lean: 63°
  friction_coefficient: 1.2
  wind_speed: 5 km/h

RACE (Carrera):
  max_lean: 65°
  friction_coefficient: 1.0
  wind_speed: 10 km/h
```

---

### Sección 5: PARÁMETROS DE ADVERSARIAL TRAINING

```yaml
# archivo: src/training/adversarial_training.py

ADVERSARIAL:
  # Generación de adversarios
  noise_type: "gaussian"          # Opciones: gaussian, uniform, laplace
  noise_scale: 0.1                # σ: [0.01, 0.5]
  perturbation_probability: 0.2   # Fracción: [0.0, 1.0]
  
  # Mezcla de datos
  adversarial_ratio: 0.2          # Fracción adversarial: [0.0, 0.5]
  adversarial_schedule: "constant"# Opciones: constant, linear, exponential
  
  # Regularización robusta
  robustness_weight: 0.1          # Factor: [0.0, 1.0]
  certified_robustness: False     # true/false
  
  # Validación
  test_noise_levels: [0, 0.05, 0.1, 0.2, 0.5]  # Pruebas
```

**Estrategias Adversariales:**

| Estrategia | Robustez | Complejidad | Tiempo |
|-----------|----------|------------|--------|
| **None** | Bajo | Bajo | Rápido |
| **Gaussian** | Medio | Medio | Normal |
| **PGD** | Muy Alto | Alto | Lento |
| **TRADES** | Equilibrado | Muy Alto | Muy Lento |

---

### Sección 6: PARÁMETROS DE VISUALIZACIÓN

```yaml
# archivo: src/visualization/bio_dashboard.py

VISUALIZATION:
  # General
  theme: "darkgrid"               # Opciones: darkgrid, whitegrid, dark, white
  dpi: 300                        # Resolución: [100, 600]
  figsize: [14, 8]                # Tamaño: varios
  
  # Paneles
  show_ecg: True
  show_hrv: True
  show_stress: True
  show_performance: True
  show_comparison: True
  
  # Colores
  color_palette: "husl"           # Opciones: husl, pastel, Set2, etc.
  accent_color: "#FF6B6B"         # Rojo
  
  # Exportación
  export_format: "png"            # Opciones: png, pdf, svg, jpg
  save_interactive: False         # Crear HTML interactivo
```

---

### Sección 7: PARÁMETROS DE DEPLOYMENT

```yaml
# archivo: src/deployment/export_to_edge.py

DEPLOYMENT:
  # Compresión
  quantization: "fp32"            # Opciones: fp32, fp16, int8, int4
  pruning_ratio: 0.0              # Fracción de pesos a eliminar: [0, 0.9]
  distillation: False             # Knowledge distillation
  
  # Optimización
  optimize_for_latency: True
  target_latency: 50              # ms: [10, 200]
  batch_size_inference: 1         # Para edge
  
  # Hardware target
  target_platform: "edge"         # Opciones: cloud, edge, mobile, embedded
  device_type: "CPU"              # Opciones: CPU, GPU, TPU, NPU
  
  # Monitoreo
  enable_profiling: True
  log_inference_time: True
  track_memory_usage: True
```

**Perfiles de Deployment:**

```
CLOUD (Servidor potente):
  quantization: fp32
  batch_size: 32
  target_latency: 5ms

EDGE (Dispositivo local):
  quantization: fp16
  batch_size: 8
  target_latency: 50ms

MOBILE (Smartphone):
  quantization: int8
  batch_size: 1
  target_latency: 200ms

EMBEDDED (Hardware integrado):
  quantization: int4
  batch_size: 1
  target_latency: 500ms
```

---

## 🎯 CONFIGURACIONES RECOMENDADAS POR CASO DE USO

### 1️⃣ Desarrollo / Testing
```python
# quick_dev_config.py
TRAINING = {
    'total_timesteps': 10_000,
    'learning_rate': 5e-4,
    'batch_size': 32
}

SIMULATION = {
    'max_speed': 150,
    'wind_speed': 0
}

ADVERSARIAL = {
    'enabled': False
}

DEPLOY = {
    'quantization': 'fp32',
    'target_latency': 100
}
```

### 2️⃣ Competición
```python
# competition_config.py
TRAINING = {
    'total_timesteps': 1_000_000,
    'learning_rate': 3e-4,
    'batch_size': 64
}

BIOMETRICS = {
    'stress_scale_factor': 1.0,
    'sampling_rate': 250
}

BIO_GATING = {
    'activation_mode': 'soft',
    'stress_threshold': 0.6
}

ADVERSARIAL = {
    'enabled': True,
    'noise_scale': 0.15
}
```

### 3️⃣ Producción
```python
# production_config.py
TRAINING = {
    'total_timesteps': 10_000_000,
    'learning_rate': 1e-4,
    'batch_size': 128,
    'clip_range': 0.1
}

BIO_GATING = {
    'activation_mode': 'adaptive',
    'stress_threshold': 0.7
}

ADVERSARIAL = {
    'enabled': True,
    'noise_scale': 0.2,
    'certified_robustness': True
}

DEPLOY = {
    'quantization': 'int8',
    'target_latency': 50,
    'enable_profiling': True
}
```

---

## 📝 CÓMO PERSONALIZAR EL SISTEMA

### Paso 1: Seleccionar Configuración Base
```bash
# Copiar configuración preestablecida
cp configs/production_config.yaml configs/mi_config.yaml
```

### Paso 2: Editar Parámetros
```yaml
# configs/mi_config.yaml
TRAINING:
  learning_rate: 2e-4  # Cambiar tasa de aprendizaje
  batch_size: 96       # Ajustar tamaño de batch

BIO_GATING:
  stress_threshold: 0.65  # Personalizar umbral
```

### Paso 3: Validar Configuración
```bash
python3 validate_config.py --config configs/mi_config.yaml
```

### Paso 4: Ejecutar con Configuración Personalizada
```bash
python3 train.py --config configs/mi_config.yaml
```

### Paso 5: Monitorear Resultados
```bash
tensorboard --logdir ./logs/
```

---

## 🔍 PARÁMETROS CRÍTICOS POR MÉTRICA

### Para Maximizar SEGURIDAD
```yaml
BIO_GATING:
  stress_threshold: 0.5  # Más sensible
  activation_mode: "hard"
  max_throttle_stressed: 0.5
  
ADVERSARIAL:
  noise_scale: 0.3  # Más robusto
```

### Para Maximizar RENDIMIENTO
```yaml
TRAINING:
  learning_rate: 5e-4  # Más rápido
  batch_size: 128
  clip_range: 0.3
  
BIO_GATING:
  stress_threshold: 0.8  # Menos restrictivo
```

### Para Equilibrio (RECOMENDADO)
```yaml
# Configuración optimizada
TRAINING:
  learning_rate: 3e-4
  batch_size: 64
  
BIO_GATING:
  activation_mode: "soft"
  stress_threshold: 0.7
  
ADVERSARIAL:
  enabled: True
  noise_scale: 0.15
```

---

## 🛠️ HERRAMIENTAS DE CONFIGURACIÓN

### Herramienta 1: Config Validator
```bash
python3 tools/validate_config.py --config configs/mi_config.yaml
```

### Herramienta 2: Parameter Sweep
```bash
python3 tools/parameter_sweep.py \
  --param learning_rate \
  --values 1e-4 3e-4 5e-4 1e-3 \
  --config configs/base_config.yaml
```

### Herramienta 3: Hyperparameter Optimizer
```bash
python3 tools/hpo.py \
  --algorithm optuna \
  --n_trials 100 \
  --metric safety_score
```

### Herramienta 4: Config Comparison
```bash
python3 tools/compare_configs.py \
  --config1 configs/competition_config.yaml \
  --config2 configs/production_config.yaml
```

---

## 📊 MATRIZ DE SENSIBILIDAD

```
Parámetro              │ Impacto en    │ Impacto en    │ Impacto en
                       │ Rendimiento   │ Seguridad     │ Latencia
───────────────────────┼───────────────┼───────────────┼──────────
learning_rate          │ ▓▓▓▓▓ Muy Alto│ ▓▓ Bajo      │ ▓ Muy Bajo
batch_size             │ ▓▓▓▓ Alto     │ ▓▓▓ Medio    │ ▓▓▓ Medio
stress_threshold       │ ▓▓ Bajo       │ ▓▓▓▓▓ Crítico│ ▓ Muy Bajo
noise_scale            │ ▓▓ Bajo       │ ▓▓▓▓ Muy Alto│ ▓ Muy Bajo
quantization           │ ▓▓ Bajo       │ ▓▓ Bajo      │ ▓▓▓▓ Crítico
```

---

## ⚡ QUICK START - CAMBIOS RÁPIDOS

### Cambio Rápido #1: Modo Entrenamiento Rápido
```bash
python3 QUICK_CONFIG.py --mode fast_training
# Resultado: entrena en 30% del tiempo, 15% menos precisión
```

### Cambio Rápido #2: Modo Máxima Seguridad
```bash
python3 QUICK_CONFIG.py --mode max_safety
# Resultado: mejor seguridad, 10% menos rendimiento
```

### Cambio Rápido #3: Modo Competición
```bash
python3 QUICK_CONFIG.py --mode competition
# Resultado: balance óptimo para carrera
```

### Cambio Rápido #4: Modo Edge Device
```bash
python3 QUICK_CONFIG.py --mode edge_device
# Resultado: latencia <50ms, cuantización int8
```

---

## 🎓 GUÍA DE AJUSTE ITERATIVO

### Ciclo de Optimización
```
1. Baseline (Medir)
   ↓
2. Hipótesis (Qué cambiar)
   ↓
3. Experimento (Hacer cambio)
   ↓
4. Evaluación (Medir resultados)
   ↓
5. Decisión (Mantener o revertir)
   ↓
6. Siguiente parámetro
```

### Ejemplo: Optimizar para Seguridad
```
Iteración 1:
  Cambio: stress_threshold: 0.7 → 0.6
  Resultado: Seguridad +8%, Rendimiento -5%
  Decisión: MANTENER

Iteración 2:
  Cambio: activation_mode: soft → hard
  Resultado: Seguridad +3%, Rendimiento -10%
  Decisión: REVERTIR (trade-off negativo)

Iteración 3:
  Cambio: noise_scale: 0.15 → 0.2
  Resultado: Robustez +12%, Rendimiento -2%
  Decisión: MANTENER
```

---

## 📞 SOPORTE DE CONFIGURACIÓN

- **Cambios simples:** Editar YAML directamente
- **Cambios complejos:** Usar herramientas HPO
- **Debugging:** Ejecutar `validate_config.py`
- **Comparación:** Usar `compare_configs.py`

**Tiempo de ajuste típico:** 2-5 iteraciones (2-4 horas)

---

**Última actualización:** 17 Enero 2026  
**Estado:** ✅ LISTO PARA PERSONALIZACIÓN

