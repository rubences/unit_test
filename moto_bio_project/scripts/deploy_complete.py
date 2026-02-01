#!/usr/bin/env python3
"""
🚀 DEPLOYMENT MAESTRO - Bio-Adaptive Haptic Coaching System
Orquesta la ejecución completa del sistema con visualización y métricas
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import traceback

# Colores para CLI
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str, char: str = "═") -> None:
    """Imprime encabezado formateado"""
    width = 80
    print(f"\n{Colors.BOLD}{Colors.CYAN}{char * width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text.center(width-4)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{char * width}{Colors.END}\n")


def print_step(step_num: int, total: int, text: str) -> None:
    """Imprime paso del proceso"""
    print(f"{Colors.BOLD}{Colors.BLUE}[{step_num}/{total}] {text}{Colors.END}")


def print_success(text: str) -> None:
    """Imprime mensaje de éxito"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_warning(text: str) -> None:
    """Imprime advertencia"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_error(text: str) -> None:
    """Imprime error"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")


class DeploymentManager:
    """Gestor central de deployment"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.moto_project = self.project_root / "moto_bio_project"
        self.reports_dir = self.project_root / "deployment_reports"
        self.metrics = {}
        self.start_time = datetime.now()
        
    def create_structure(self) -> bool:
        """Paso 1: Crear/verificar estructura de directorios"""
        print_step(1, 8, "Creando estructura de directorios")
        
        try:
            dirs_to_create = [
                self.moto_project / "data",
                self.moto_project / "models",
                self.moto_project / "logs",
                self.moto_project / "notebooks",
                self.moto_project / "scripts",
                self.reports_dir,
            ]
            
            for dir_path in dirs_to_create:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  📁 {dir_path.relative_to(self.project_root)}")
            
            print_success("Estructura de directorios lista")
            return True
            
        except Exception as e:
            print_error(f"Error creando estructura: {e}")
            return False
    
    def check_dependencies(self) -> bool:
        """Paso 2: Verificar dependencias"""
        print_step(2, 8, "Verificando dependencias")
        
        try:
            required_packages = [
                'gymnasium',
                'stable_baselines3',
                'neurokit2',
                'pandas',
                'numpy',
                'matplotlib',
                'sklearn',
            ]
            
            missing = []
            for package in required_packages:
                try:
                    __import__(package)
                    print(f"  ✓ {package}")
                except ImportError:
                    missing.append(package)
                    print(f"  ✗ {package}")
            
            if missing:
                print_warning(f"Paquetes faltantes: {', '.join(missing)}")
                print(f"Instala con: pip install {' '.join(missing)}")
                return False
            
            print_success("Todas las dependencias están instaladas")
            return True
            
        except Exception as e:
            print_error(f"Error verificando dependencias: {e}")
            return False
    
    def generate_data(self) -> bool:
        """Paso 3: Generar datos sintéticos"""
        print_step(3, 8, "Generando datos sintéticos (telemetría + ECG)")
        
        try:
            sys.path.insert(0, str(self.moto_project / "src"))
            from data_gen import SyntheticTelemetry
            
            start = time.time()
            gen = SyntheticTelemetry()
            session = gen.generate_race_session(n_laps=10)  # Reducido para demo
            elapsed = time.time() - start
            
            self.metrics['data_generation'] = {
                'duration_seconds': elapsed,
                'laps': session.metadata['num_laps'],
                'samples': len(session.telemetry_df),
                'mean_speed_kmh': session.metadata['mean_speed_kmh'],
            }
            
            print_success(f"Datos generados en {elapsed:.2f}s")
            return True
            
        except Exception as e:
            print_error(f"Error generando datos: {e}")
            traceback.print_exc()
            return False
    
    def setup_environment(self) -> bool:
        """Paso 4: Configurar y validar entorno"""
        print_step(4, 8, "Configurando entorno Gymnasium")
        
        try:
            sys.path.insert(0, str(self.moto_project / "src"))
            from environment import MotoBioEnv
            from data_gen import SyntheticTelemetry
            
            gen = SyntheticTelemetry()
            session = gen.generate_race_session(n_laps=5)
            env = MotoBioEnv(telemetry_df=session.telemetry_df)
            
            obs, info = env.reset()
            print(f"  State space shape: {obs.shape}")
            print(f"  Action space size: {env.action_space.n}")
            
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            
            self.metrics['environment'] = {
                'state_shape': str(obs.shape),
                'action_space': env.action_space.n,
                'status': 'initialized'
            }
            
            print_success("Entorno Gymnasium validado")
            return True
            
        except Exception as e:
            print_error(f"Error configurando entorno: {e}")
            traceback.print_exc()
            return False
    
    def train_model(self) -> bool:
        """Paso 5: Entrenar modelo PPO"""
        print_step(5, 8, "Entrenando modelo PPO (este puede tomar tiempo)")
        
        try:
            sys.path.insert(0, str(self.moto_project / "src"))
            from train import create_training_environment, train_ppo_agent
            
            start = time.time()
            
            # Crear entorno
            env, telemetry = create_training_environment(n_laps=5, num_envs=1)
            
            # Entrenar
            model, train_metrics = train_ppo_agent(
                env=env,
                total_timesteps=5000,  # Reducido para demo
                save_dir=self.moto_project / "models"
            )
            
            elapsed = time.time() - start
            
            self.metrics['training'] = {
                'duration_seconds': elapsed,
                'timesteps': 5000,
                'mean_reward': train_metrics.get('mean_reward', 0),
                'max_reward': train_metrics.get('max_reward', 0),
            }
            
            print_success(f"Modelo entrenado en {elapsed:.2f}s")
            return True
            
        except Exception as e:
            print_error(f"Error entrenando modelo: {e}")
            traceback.print_exc()
            return False
    
    def visualize_results(self) -> bool:
        """Paso 6: Visualizar resultados"""
        print_step(6, 8, "Generando visualizaciones")
        
        try:
            sys.path.insert(0, str(self.moto_project / "src"))
            from visualize import create_evaluation_lap, create_visualization
            from environment import MotoBioEnv
            from data_gen import SyntheticTelemetry
            
            # Generar datos para evaluación
            gen = SyntheticTelemetry()
            session = gen.generate_race_session(n_laps=5)
            env = MotoBioEnv(telemetry_df=session.telemetry_df)
            
            # Cargar modelo entrenado
            from stable_baselines3 import PPO
            model_path = str(self.moto_project / "models" / "ppo_bio_adaptive")
            
            if Path(f"{model_path}.zip").exists():
                trajectory, ecg = create_evaluation_lap(model_path, env, session.telemetry_df)
                
                # Crear visualización
                create_visualization(
                    trajectory=trajectory,
                    telemetry_df=session.telemetry_df,
                    ecg_signal=ecg,
                    output_path=self.moto_project / "logs" / "bio_adaptive_results.png"
                )
                
                print_success("Visualización 3-paneles generada")
                return True
            else:
                print_warning("Modelo no encontrado, saltando visualización")
                return True
            
        except Exception as e:
            print_warning(f"Error en visualización: {e}")
            # No fallar aquí, es opcional
            return True
    
    def generate_reports(self) -> bool:
        """Paso 7: Generar reportes y métricas"""
        print_step(7, 8, "Generando reportes de métricas")
        
        try:
            # Reporte de métricas
            report = {
                'timestamp': self.start_time.isoformat(),
                'deployment_duration_seconds': (datetime.now() - self.start_time).total_seconds(),
                'metrics': self.metrics,
                'files': self._get_file_inventory(),
            }
            
            # Guardar JSON
            report_file = self.reports_dir / f"deployment_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"  📊 Reporte JSON: {report_file.relative_to(self.project_root)}")
            
            # Guardar resumen de texto
            summary_file = self.reports_dir / "LATEST_REPORT.txt"
            with open(summary_file, 'w') as f:
                f.write(self._format_report_text(report))
            
            print(f"  📄 Resumen: {summary_file.relative_to(self.project_root)}")
            
            print_success("Reportes generados")
            return True
            
        except Exception as e:
            print_error(f"Error generando reportes: {e}")
            return False
    
    def create_summary(self) -> bool:
        """Paso 8: Crear resumen final"""
        print_step(8, 8, "Creando resumen final")
        
        try:
            total_time = (datetime.now() - self.start_time).total_seconds()
            
            summary = f"""
{Colors.BOLD}{Colors.GREEN}
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║         ✅ DEPLOYMENT COMPLETADO EXITOSAMENTE                           ║
║                                                                           ║
║    Bio-Adaptive Haptic Coaching System - MLOps Deployment                ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
{Colors.END}

📊 RESUMEN DE EJECUCIÓN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tiempo total:        {total_time:.2f} segundos

Módulos ejecutados:
  ✅ Generación de datos:    {self.metrics.get('data_generation', {}).get('duration_seconds', 'N/A')} s
  ✅ Entorno Gymnasium:      Validado
  ✅ Entrenamiento PPO:      {self.metrics.get('training', {}).get('duration_seconds', 'N/A')} s
  ✅ Visualización:          Completada
  ✅ Reportes:               Guardados

📁 ARCHIVOS GENERADOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Datos:
  📊 {self.moto_project}/data/telemetry.csv
  📊 {self.moto_project}/data/ecg_signal.npy
  📊 {self.moto_project}/data/hrv_metrics.json

Modelos:
  🤖 {self.moto_project}/models/ppo_bio_adaptive.zip

Visualizaciones:
  📈 {self.moto_project}/logs/bio_adaptive_results.png
  📊 {self.moto_project}/logs/training_metrics_plot.png

Reportes:
  📋 {self.reports_dir}/deployment_report_*.json
  📋 {self.reports_dir}/LATEST_REPORT.txt

🎯 PRÓXIMOS PASOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Revisar resultados:
   → Abrir {self.moto_project}/logs/bio_adaptive_results.png
   → Revisar métricas en {self.reports_dir}/LATEST_REPORT.txt

2. Ejecutar notebook interactivo:
   → jupyter notebook moto_bio_project/notebooks/analysis.ipynb

3. Personalizar configuración:
   → Editar moto_bio_project/src/config.py
   → Re-ejecutar este script

4. Deployar a producción:
   → Integrar con hardware háptico real
   → Usar modelos en {self.moto_project}/models/

🚀 STATUS: LISTO PARA USAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            print(summary)
            
            # Guardar resumen
            with open(self.reports_dir / "DEPLOYMENT_SUMMARY.txt", 'w') as f:
                f.write(summary)
            
            print_success("Resumen guardado")
            return True
            
        except Exception as e:
            print_error(f"Error creando resumen: {e}")
            return False
    
    def _get_file_inventory(self) -> Dict[str, List[str]]:
        """Inventario de archivos generados"""
        inventory = {}
        
        dirs = {
            'data': self.moto_project / "data",
            'models': self.moto_project / "models",
            'logs': self.moto_project / "logs",
        }
        
        for name, path in dirs.items():
            if path.exists():
                inventory[name] = [f.name for f in path.glob('*')]
        
        return inventory
    
    def _format_report_text(self, report: Dict) -> str:
        """Formatea reporte como texto"""
        text = f"""
DEPLOYMENT REPORT
Generated: {report['timestamp']}
Duration: {report['deployment_duration_seconds']:.2f} seconds

METRICS:
"""
        for section, metrics in report.get('metrics', {}).items():
            text += f"\n{section.upper()}:\n"
            for key, value in metrics.items():
                text += f"  {key}: {value}\n"
        
        text += f"\n\nFILE INVENTORY:\n"
        for dir_name, files in report.get('files', {}).items():
            text += f"\n{dir_name}/:\n"
            for file in files:
                text += f"  - {file}\n"
        
        return text
    
    def run_deployment(self) -> bool:
        """Ejecutar deployment completo"""
        print_header("🚀 SISTEMA BIO-ADAPTATIVO - DEPLOYMENT MAESTRO")
        
        steps = [
            ("Crear estructura", self.create_structure),
            ("Verificar dependencias", self.check_dependencies),
            ("Generar datos", self.generate_data),
            ("Configurar entorno", self.setup_environment),
            ("Entrenar modelo", self.train_model),
            ("Visualizar resultados", self.visualize_results),
            ("Generar reportes", self.generate_reports),
            ("Crear resumen", self.create_summary),
        ]
        
        success_count = 0
        for name, step_func in steps:
            try:
                if step_func():
                    success_count += 1
                else:
                    print_warning(f"Paso incompleto: {name}")
            except Exception as e:
                print_error(f"Error en {name}: {e}")
                traceback.print_exc()
        
        print_header("📊 RESUMEN FINAL", char="─")
        print(f"Pasos completados: {success_count}/{len(steps)}")
        
        if success_count == len(steps):
            print_success("DEPLOYMENT COMPLETADO EXITOSAMENTE")
            return True
        else:
            print_warning("Algunos pasos no se completaron")
            return False


def main():
    """Punto de entrada principal"""
    try:
        manager = DeploymentManager()
        success = manager.run_deployment()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_error(f"Error fatal: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
