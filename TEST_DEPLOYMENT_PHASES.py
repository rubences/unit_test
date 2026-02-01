#!/usr/bin/env python3
"""
Test rápido de las fases problemáticas del MASTER_DEPLOYMENT
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
root_dir = Path(__file__).parent
moto_src_dir = root_dir / "moto_bio_project" / "src"
sys.path.insert(0, str(moto_src_dir))

print("=" * 70)
print("TEST DE FASES PROBLEMÁTICAS")
print("=" * 70)

# ============================================================================
# FASE 5: GENERACIÓN DE DATOS
# ============================================================================
print("\n[FASE 5] GENERACIÓN DE DATOS SINTÉTICOS\n")

try:
    from data_gen import SyntheticTelemetry
    
    print("⏳ Generando 2 laps de telemetría...")
    gen = SyntheticTelemetry()
    session = gen.generate_race_session(n_laps=2)
    
    print(f"✅ Datos generados exitosamente")
    print(f"  • Muestras: {len(session.telemetry_df)}")
    print(f"  • Velocidad media: {session.metadata.get('mean_speed_kmh', 0):.1f} km/h")
    
    # Guardar datos temporales
    temp_data_dir = root_dir / "temp_test_data"
    temp_data_dir.mkdir(exist_ok=True)
    data_file = temp_data_dir / "test_telemetry.csv"
    session.telemetry_df.to_csv(data_file, index=False)
    print(f"  • Archivo: {data_file}")
    
    telemetry_df = session.telemetry_df
    phase5_success = True
    
except Exception as e:
    print(f"❌ Error en generación: {e}")
    import traceback
    traceback.print_exc()
    phase5_success = False
    telemetry_df = None

# ============================================================================
# FASE 6: ENTRENAMIENTO DE MODELOS
# ============================================================================
print("\n[FASE 6] ENTRENAMIENTO DE MODELOS (mini-training)\n")

if phase5_success and telemetry_df is not None:
    try:
        from train import create_training_environment, train_ppo_agent
        from environment import MotoBioEnv
        
        print("⏳ Creando entorno de entrenamiento...")
        env = MotoBioEnv(telemetry_df=telemetry_df)
        
        print("⏳ Entrenando PPO (100 steps de prueba)...")
        model, training_metrics = train_ppo_agent(
            env=env,
            total_timesteps=100,  # Solo 100 steps para test rápido
            save_freq=50,
            model_name="test_model"
        )
        
        print(f"✅ Entrenamiento completado")
        print(f"  • Episodios: {training_metrics.get('num_episodes', 0)}")
        print(f"  • Recompensa final: {training_metrics.get('final_episode_reward', 0):.2f}")
        
        # Guardar modelo temporal
        temp_model_dir = root_dir / "temp_test_models"
        temp_model_dir.mkdir(exist_ok=True)
        model_path = temp_model_dir / "test_ppo_model.zip"
        model.save(str(model_path))
        print(f"  • Modelo: {model_path}")
        
        phase6_success = True
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        phase6_success = False
        model_path = None
else:
    print("⚠️  Saltando FASE 6 (FASE 5 falló)")
    phase6_success = False
    model_path = None

# ============================================================================
# FASE 7: EVALUACIÓN DE MODELOS
# ============================================================================
print("\n[FASE 7] EVALUACIÓN DE MODELOS\n")

if phase6_success and model_path and model_path.exists():
    try:
        from environment import MotoBioEnv
        from stable_baselines3 import PPO
        import numpy as np
        
        print("⏳ Cargando modelo entrenado...")
        model = PPO.load(str(model_path))
        
        print("⏳ Ejecutando 2 episodios de evaluación...")
        env = MotoBioEnv(telemetry_df=telemetry_df)
        
        rewards = []
        for ep in range(2):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 100:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                steps += 1
            
            rewards.append(episode_reward)
            print(f"  • Episodio {ep+1}: Recompensa = {episode_reward:.2f}")
        
        print(f"\n✅ Evaluación completada")
        print(f"  • Recompensa media: {np.mean(rewards):.2f}")
        print(f"  • Recompensa máx: {np.max(rewards):.2f}")
        
        phase7_success = True
        
    except Exception as e:
        print(f"❌ Error en evaluación: {e}")
        import traceback
        traceback.print_exc()
        phase7_success = False
else:
    print("⚠️  Modelo no encontrado (FASE 6 no completada)")
    phase7_success = False

# ============================================================================
# RESUMEN
# ============================================================================
print("\n" + "=" * 70)
print("RESUMEN DE TESTS")
print("=" * 70)

results = {
    "FASE 5 (Generación de datos)": "✅ PASS" if phase5_success else "❌ FAIL",
    "FASE 6 (Entrenamiento)": "✅ PASS" if phase6_success else "❌ FAIL",
    "FASE 7 (Evaluación)": "✅ PASS" if phase7_success else "❌ FAIL"
}

for fase, status in results.items():
    print(f"{fase}: {status}")

# Cleanup
try:
    import shutil
    if (root_dir / "temp_test_data").exists():
        shutil.rmtree(root_dir / "temp_test_data")
    if (root_dir / "temp_test_models").exists():
        shutil.rmtree(root_dir / "temp_test_models")
    print("\n✓ Archivos temporales limpiados")
except:
    pass

all_success = phase5_success and phase6_success and phase7_success

print("\n" + "=" * 70)
if all_success:
    print("✅ TODOS LOS TESTS PASARON - FIXES VERIFICADOS")
else:
    print("⚠️  ALGUNOS TESTS FALLARON - REVISAR ERRORES ARRIBA")
print("=" * 70)

sys.exit(0 if all_success else 1)
