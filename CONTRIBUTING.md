# Contribuir a Moto-Edge-RL

Gracias por tu interés. Sigue estos pasos para reproducir experimentos y proponer mejoras.

## Código de Conducta
Al participar aceptas el [Code of Conduct](CODE_OF_CONDUCT.md).

## Flujo básico
1) Clona el repositorio:
```bash
git clone https://github.com/rubences/Coaching-for-Competitive-Motorcycle-Racing.git
cd Coaching-for-Competitive-Motorcycle-Racing
```

2) Crea un entorno virtual y activa:
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

3) Instala dependencias y extras de desarrollo:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .[dev]
```

4) Verifica calidad y pruebas locales:
```bash
flake8 . --max-line-length=120 --extend-ignore=E203,W503
pytest
```

5) Smoke test rápido de entrenamiento (100 pasos PPO):
```bash
python - <<'PY'
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from simulation.motorcycle_env import MotorcycleEnv

env = make_vec_env(MotorcycleEnv, n_envs=1, env_kwargs={"render_mode": None})
model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, learning_rate=3e-4, verbose=0)
model.learn(total_timesteps=100)
env.close()
PY
```

## Rama y commits
- Usa ramas descriptivas: `feature/xxx`, `fix/xxx`, `docs/xxx`.
- Commits claros y pequeños. Incluye contexto en el mensaje.

## Antes de abrir un PR
- Asegúrate de que flake8 y pytest pasan.
- Añade/actualiza tests cuando cambies funcionalidad.
- Actualiza la documentación relevante (README, docs/, ejemplos).
- Describe reproducibilidad (datos, seeds, comandos).

## Reproducir experimentos
- Configuración principal en `configs/`.
- Scripts en `scripts/` y `src/moto_edge_rl/`.
- Usa `pytest` para validar cambios en entornos y agentes.
- Para datasets Minari/HDF5, verifica rutas en `configs/train_config.yaml` o variables de entorno.

## Soporte
Si algo falla, abre un issue con: comando ejecutado, salida completa y versión de Python.
