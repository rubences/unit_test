#!/usr/bin/env python3
"""
CLI Central - Interfaz unificada para todo el sistema
Permite ejecutar: entrenar, desplegar, analizar, visualizar desde un único punto
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# ANSI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class SystemManager:
    """Gestor central del sistema"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent.parent
        self.system_dir = self.root_dir / "system"
        self.workspace_dir = self.root_dir / "workspace"
        self.config_file = self.system_dir / "config" / "system.json"
        self.load_config()
        
    def load_config(self):
        """Cargar configuración del sistema"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()
            self.save_config()
    
    def _default_config(self) -> Dict:
        """Configuración por defecto"""
        return {
            "version": "1.0.0",
            "environment": "development",
            # Compatibilidad legado
            "training": {
                "algorithm": "PPO",
                "episodes": 5,
                "learning_rate": 0.0003,
                "batch_size": 64,
                "gamma": 0.99
            },
            # Nueva estructura
            "components": {
                "reinforcement_learning": {
                    "algorithm": "PPO",
                    "episodes": 5,
                    "learning_rate": 0.0003,
                    "batch_size": 64,
                    "gamma": 0.99
                },
                "simulation": {
                    "environment": "motorcycle_env",
                    "max_velocity": 300,
                    "timesteps": 300
                },
                "biometrics": {
                    "sampling_rate": 250,
                    "signals": ["ecg", "hr", "hrv"]
                },
                "adversarial_training": {
                    "enabled": False,
                    "noise_levels": 50,
                    "max_noise_scale": 0.5
                },
                "safety": {
                    "bio_gating": True,
                    "stress_threshold": 0.7,
                    "activation_mode": "adaptive"
                }
            },
            "deployment": {
                "target": "local",
                "quantization": "fp32",
                "timeout": 30,
                "monitoring": True,
                "auto_rollback": True
            },
            "visualization": {
                "dpi": 300,
                "format": "png",
                "interactive": True,
                "server_port": 7860,
                "mode": "html",
                "theme": "dark"
            }
        }
    
    def save_config(self):
        """Guardar configuración"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def apply_preset(self, preset: str):
        """Aplicar presets de configuración: 'fast' (rápido) o 'robust' (robusto)"""
        preset = (preset or "").strip().lower()
        cfg = self.config
        rl = cfg.get("components", {}).get("reinforcement_learning")
        sim = cfg.get("components", {}).get("simulation")
        adv = cfg.get("components", {}).get("adversarial_training")
        saf = cfg.get("components", {}).get("safety")

        if preset in ("1", "fast", "rapido", "rápido"):
            if rl:
                rl["episodes"] = 10
                rl["learning_rate"] = 0.0005
                rl["batch_size"] = 64
                rl["gamma"] = 0.98
            if sim:
                sim["timesteps"] = 200
            if adv:
                adv["enabled"] = False
            if saf:
                saf["stress_threshold"] = 0.7
            return "Entrenamiento rápido aplicado"

        if preset in ("2", "robust", "robusto"):
            if rl:
                rl["episodes"] = 50
                rl["learning_rate"] = 0.0002
                rl["batch_size"] = 128
                rl["gamma"] = 0.995
            if sim:
                sim["timesteps"] = 600
            if adv:
                adv["enabled"] = True
                adv["noise_levels"] = 100
                adv["max_noise_scale"] = 0.7
            if saf:
                saf["stress_threshold"] = 0.6
                saf["activation_mode"] = "adaptive"
            return "Entrenamiento robusto aplicado"

        return "Sin cambios"
    
    def print_banner(self):
        """Banner de bienvenida"""
        banner = f"""
{Colors.BOLD}{Colors.CYAN}
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🏍️  SISTEMA DE COACHING BIO-ADAPTATIVO                       ║
║   Motociclismo de Competencia con Retroalimentación Táctil     ║
║                                                                  ║
║   {Colors.GREEN}✓ Versión {self.config['version']}{Colors.CYAN}                                    ║
║   ✓ Estado: {Colors.GREEN}OPERATIVO{Colors.CYAN}                                ║
║   ✓ Componentes: 37 módulos integrados                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
{Colors.ENDC}
"""
        print(banner)
    
    def print_menu(self):
        """Menú principal"""
        menu = f"""
{Colors.BOLD}{Colors.CYAN}MENÚ PRINCIPAL{Colors.ENDC}
{Colors.YELLOW}────────────────────────────────────────────────────────────────{Colors.ENDC}

{Colors.GREEN}1. 🎯 ENTRENAR{Colors.ENDC}
   • Ejecutar algoritmo PPO con biométricos
   • Generar modelos entrenados
   • Monitorear convergencia

{Colors.GREEN}2. 🚀 DESPLEGAR{Colors.ENDC}
   • Despliegue en producción
   • Blue-green deployment
   • Validación de salud

{Colors.GREEN}3. 📊 ANALIZAR{Colors.ENDC}
   • Análisis de resultados
   • Reportes detallados
   • Métricas de rendimiento

{Colors.GREEN}4. 🎨 VISUALIZAR{Colors.ENDC}
   • Dashboard interactivo
   • Gráficos de entrenamientoKpis en tiempo real

{Colors.GREEN}5. ⚙️ CONFIGURAR{Colors.ENDC}
   • Parámetros de entrenamiento
   • Configuración de despliegue
   • Personalización del sistema

{Colors.GREEN}6. 🧪 EJECUTAR DEMOS{Colors.ENDC}
   • Demo biométrica
   • Demo RL
   • Demo simulación
   • Demo adversarial
   • Demo comparación

{Colors.GREEN}7. 📚 DOCUMENTACIÓN{Colors.ENDC}
   • Guías de uso
   • Referencia de APIs
   • Ejemplos

{Colors.GREEN}0. 🚪 SALIR{Colors.ENDC}

{Colors.YELLOW}────────────────────────────────────────────────────────────────{Colors.ENDC}
"""
        print(menu)
    
    def train(self, episodes: int = None, algorithm: str = None):
        """Ejecutar entrenamiento"""
        rl_config = self.config['components']['reinforcement_learning']
        print(f"\n{Colors.BOLD}{Colors.BLUE}[ENTRENAMIENTO]{Colors.ENDC}")
        print(f"  • Algoritmo: {algorithm or rl_config['algorithm']}")
        print(f"  • Episodios: {episodes or rl_config['episodes']}")
        print(f"  • Tasa aprendizaje: {rl_config['learning_rate']}")
        print(f"  • Tamaño batch: {rl_config['batch_size']}\n")
        
        # Ejecutar entrenamiento
        cmd = [
            "python3", str(self.root_dir / "INTERACTIVE_DEMOS.py"),
            "--mode", "train",
            "--episodes", str(episodes or rl_config['episodes'])
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"\n{Colors.GREEN}✓ Entrenamiento completado{Colors.ENDC}\n")
        except subprocess.CalledProcessError as e:
            print(f"\n{Colors.RED}✗ Error en entrenamiento: {e}{Colors.ENDC}\n")
    
    def deploy(self, target: str = None):
        """Desplegar sistema"""
        target = target or self.config['deployment']['target']
        print(f"\n{Colors.BOLD}{Colors.BLUE}[DESPLIEGUE]{Colors.ENDC}")
        print(f"  • Destino: {target}")
        print(f"  • Cuantización: {self.config['deployment']['quantization']}")
        print(f"  • Timeout: {self.config['deployment']['timeout']}s\n")
        
        deployment_log = self.workspace_dir / "logs" / f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        deployment_log.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python3", str(self.root_dir / "MASTER_DEPLOYMENT.py"),
            "--target", target,
            "--log", str(deployment_log)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"\n{Colors.GREEN}✓ Despliegue completado{Colors.ENDC}")
            print(f"  📋 Log: {deployment_log}\n")
        except subprocess.CalledProcessError as e:
            print(f"\n{Colors.RED}✗ Error en despliegue: {e}{Colors.ENDC}\n")
    
    def analyze(self):
        """Ejecutar análisis"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}[ANÁLISIS]{Colors.ENDC}\n")
        
        # Buscar resultados en múltiples ubicaciones
        possible_locations = [
            self.workspace_dir / "results" / "demo_results.json",
            self.root_dir / "DEPLOYMENT_ARTIFACTS" / "demo_results.json",
            self.root_dir / "demo_results.json",
            self.root_dir / "workspace" / "results" / "demo_results.json"
        ]
        
        results_file = None
        for location in possible_locations:
            if location.exists():
                results_file = location
                break
        
        if results_file:
            print(f"  📁 Archivo: {results_file.name}")
            print(f"  📍 Ubicación: {results_file.parent}\n")
            
            with open(results_file) as f:
                data = json.load(f)
            
            # Análisis Biométrico
            if 'biometric' in data:
                print(f"{Colors.GREEN}📊 MÉTRICAS BIOMÉTRICAS{Colors.ENDC}")
                bio = data['biometric']
                print(f"  • FC Media: {bio['mean_hr']:.1f} bpm")
                print(f"  • Variabilidad: {bio['std_hr']:.1f} bpm")
                if 'rmssd' in bio:
                    print(f"  • RMSSD: {bio['rmssd']:.4f}")
                print(f"  • Estrés: {bio['stress_level']:.1f}%\n")
            
            # Análisis RL
            if 'training' in data:
                print(f"{Colors.GREEN}🎯 MÉTRICAS RL (PPO){Colors.ENDC}")
                train = data['training']
                
                if 'episodes' in train:
                    print(f"  • Episodios: {train['episodes']}")
                    print(f"  • Recompensa Media: {train['mean_reward']:.2f}")
                    print(f"  • Recompensa Máx: {train['max_reward']:.2f}")
                elif 'episode_rewards' in train:
                    rewards = train['episode_rewards']
                    print(f"  • Episodios ejecutados: {len(rewards)}")
                    print(f"  • Recompensa Media: {sum(rewards)/len(rewards):.2f}")
                    print(f"  • Recompensa Máx: {max(rewards):.2f}")
                    print(f"  • Recompensa Mín: {min(rewards):.2f}")
                
                if 'learning_curve' in train:
                    curve = train['learning_curve']
                    print(f"  • Mejora total: {curve[-1] - curve[0]:.2f}")
                print()
            
            # Análisis Simulación
            if 'simulation' in data:
                print(f"{Colors.GREEN}🏁 MÉTRICAS SIMULACIÓN{Colors.ENDC}")
                sim = data['simulation']
                print(f"  • Velocidad Max: {sim['max_velocity']:.1f} km/h")
                print(f"  • Inclinación: {sim['max_lean_angle']:.1f}°")
                print(f"  • Aceleración: {sim['mean_acceleration']:.2f} m/s²")
                if 'control_smoothness' in sim:
                    print(f"  • Suavidad Control: {sim['control_smoothness']:.3f}")
                print()
            
            # Análisis Adversarial
            if 'adversarial' in data:
                print(f"{Colors.GREEN}⚔️ ROBUSTEZ ADVERSARIAL{Colors.ENDC}")
                adv = data['adversarial']
                print(f"  • Mejora: +{adv['mean_improvement']:.2f}%")
                print(f"  • Robustez Max Ruido: {adv['robustness_at_max_noise']:.2f}%")
                if 'noise_levels_tested' in adv:
                    print(f"  • Niveles de ruido: {adv['noise_levels_tested']}")
                print()
            
            # Resumen de Comparación
            if 'comparison' in data:
                print(f"{Colors.GREEN}📈 COMPARACIÓN DE CONFIGURACIONES{Colors.ENDC}")
                comp = data['comparison']
                if 'winner' in comp:
                    print(f"  • Configuración óptima: {comp['winner']}")
                if 'performance_difference' in comp:
                    print(f"  • Diferencia: {comp['performance_difference']:.2f}%")
                print()
            
            print(f"{Colors.BOLD}═══════════════════════════════════════════════{Colors.ENDC}")
            print(f"{Colors.GREEN}✓ Análisis completado{Colors.ENDC}")
            print(f"  Total de componentes analizados: {len(data)}")
            print(f"{Colors.BOLD}═══════════════════════════════════════════════{Colors.ENDC}\n")
        else:
            print(f"{Colors.YELLOW}⚠️  No hay resultados disponibles.{Colors.ENDC}")
            print(f"  Ejecuta primero: python3 main.py demos\n")
            print(f"  Buscado en:")
            for loc in possible_locations:
                print(f"    • {loc}")
            print()
    
    def visualize(self, mode: Optional[str] = None):
        """Abrir dashboard de visualización"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}[VISUALIZACIÓN]{Colors.ENDC}\n")
        
        # Determinar modo de visualización
        vis_cfg = self.config.get("visualization", {})
        chosen_mode = (mode or vis_cfg.get("mode") or "html").strip().lower()

        if chosen_mode == "gradio":
            app_path = self.root_dir / "system" / "visualization" / "gradio_app.py"
            port = vis_cfg.get("server_port", 7860)
            if not app_path.exists():
                print(f"  {Colors.RED}✗ Gradio app no encontrada en {app_path}{Colors.ENDC}\n")
                print("  Ejecuta primero la instalación o verifica el archivo.")
                return
            print(f"  🚀 Iniciando interfaz web Gradio en puerto {port}...")
            try:
                subprocess.run(["python3", str(app_path)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n{Colors.RED}✗ Error al iniciar Gradio: {e}{Colors.ENDC}\n")
            return

        # Modo HTML por defecto
        dashboard = self.root_dir / "dashboard.html"
        if dashboard.exists():
            print(f"  📊 Abriendo dashboard interactivo...")
            url = dashboard.as_uri()
            try:
                import webbrowser
                opened = webbrowser.open(url)
                if opened:
                    print(f"  {Colors.GREEN}✓ Dashboard abierto en el navegador predeterminado{Colors.ENDC}\n")
                else:
                    print(f"  {Colors.YELLOW}⚠️ No se pudo abrir automáticamente. Abre el archivo manualmente:{Colors.ENDC}")
                    print(f"    {dashboard}\n")
            except Exception as e:
                print(f"  {Colors.YELLOW}⚠️ No se pudo abrir automáticamente ({e}).{Colors.ENDC}")
                print(f"  Abre manualmente: {dashboard}\n")
        else:
            print(f"  {Colors.RED}✗ Dashboard no encontrado{Colors.ENDC}\n")
    
    def configure(self):
        """Configurar parámetros (modo fácil, sin conocimientos de programación)"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}[CONFIGURACIÓN]{Colors.ENDC}\n")

        # Presets rápidos
        print("Presets disponibles:")
        print("  1. Entrenamiento rápido (más ágil)")
        print("  2. Entrenamiento robusto (más estabilidad)")
        try:
            preset_choice = input("¿Aplicar preset? (1/2/n): ").strip().lower()
        except EOFError:
            preset_choice = "n"
        if preset_choice in ("1", "2", "fast", "rapido", "rápido", "robust", "robusto"):
            msg = self.apply_preset(preset_choice)
            self.save_config()
            print(f"\n{Colors.GREEN}✓ {msg}{Colors.ENDC}\n")

        def cast_value(current, text):
            """Convierte texto a tipo apropiado según el valor actual"""
            if text is None or text == "":
                return current
            t = type(current)
            # Normalizar booleanos
            if t is bool:
                txt = text.strip().lower()
                return txt in ("true", "1", "si", "sí", "on")
            # Números
            if t in (int, float):
                try:
                    return t(text)
                except Exception:
                    # Intentar convertir a float primero
                    try:
                        val = float(text)
                        return t(val) if t is int else val
                    except Exception:
                        return current
            # Listas separadas por coma
            if t is list:
                vals = [v.strip() for v in text.split(',') if v.strip()]
                return vals if vals else current
            # Por defecto: string
            return text

        def get_path(cfg, path):
            ref = cfg
            for k in path.split('.'):
                ref = ref[k]
            return ref

        def set_path(cfg, path, value):
            parts = path.split('.')
            ref = cfg
            for k in parts[:-1]:
                ref = ref[k]
            ref[parts[-1]] = value

        # Construir menú amigable
        rl_base = "components.reinforcement_learning"
        sim_base = "components.simulation"
        bio_base = "components.biometrics"
        adv_base = "components.adversarial_training"
        saf_base = "components.safety"

        # Ayuda contextual por parámetro (lenguaje no técnico)
        help_texts = {
            f"{rl_base}.algorithm": "Tipo de entrenador. Recomendado: PPO (equilibrado y estable).",
            f"{rl_base}.episodes": "Cuántas rondas completas de entrenamiento quieres realizar.",
            f"{rl_base}.learning_rate": "Velocidad de ajuste del modelo. Más alto aprende rápido, demasiado alto puede ser inestable.",
            f"{rl_base}.batch_size": "Cantidad de ejemplos que se usan juntos en cada actualización.",
            f"{rl_base}.gamma": "Peso del futuro frente al presente. Más alto valora más el largo plazo.",

            f"{sim_base}.environment": "Nombre del entorno de simulación.",
            f"{sim_base}.max_velocity": "Velocidad máxima de la moto (km/h) en la simulación.",
            f"{sim_base}.timesteps": "Duración del episodio en pasos.",

            f"{bio_base}.sampling_rate": "Frecuencia de muestreo del ECG (veces por segundo).",
            f"{bio_base}.signals": "Señales biométricas que se usarán (por ejemplo: ecg, hr, hrv).",

            f"{adv_base}.noise_levels": "Cantidad de niveles de ruido a probar en robustez.",
            f"{adv_base}.max_noise_scale": "Intensidad máxima del ruido (0 a 1).",

            f"{saf_base}.bio_gating": "Activa el modo seguridad bio-adaptativo (bloquea acciones si hay estrés alto).",
            f"{saf_base}.stress_threshold": "Nivel de estrés a partir del cual se activa la protección.",
            f"{saf_base}.activation_mode": "Cómo se activa la seguridad (adaptive: se ajusta sola).",

            "visualization.theme": "Apariencia del panel: dark (oscuro) o light (claro).",
            "visualization.server_port": "Puerto del servidor para abrir el panel.",
            "visualization.dpi": "Resolución de imágenes (puntos por pulgada).",
            "visualization.format": "Formato de imagen (por ejemplo: png).",
            "visualization.interactive": "Si el panel será interactivo.",

            "deployment.target": "Dónde se despliega: local (tu equipo) o production (servidor).",
            "deployment.quantization": "Formato numérico: fp32 (preciso) o int8 (rápido).",
            "deployment.timeout": "Tiempo máximo de espera (segundos) antes de dar por fallido.",
            "deployment.monitoring": "Activa el seguimiento del estado tras desplegar.",
            "deployment.auto_rollback": "Vuelve a la versión anterior si algo sale mal.",
        }

        menu = [
            {
                "title": "Entrenamiento (PPO)",
                "items": [
                    ("Algoritmo", f"{rl_base}.algorithm"),
                    ("Episodios", f"{rl_base}.episodes"),
                    ("Tasa de aprendizaje", f"{rl_base}.learning_rate"),
                    ("Tamaño de batch", f"{rl_base}.batch_size"),
                    ("Gamma (descuento)", f"{rl_base}.gamma"),
                ],
            },
            {
                "title": "Simulación", 
                "items": [
                    ("Entorno", f"{sim_base}.environment"),
                    ("Velocidad máxima (km/h)", f"{sim_base}.max_velocity"),
                    ("Pasos por episodio", f"{sim_base}.timesteps"),
                ],
            },
            {
                "title": "Biométricos", 
                "items": [
                    ("Muestreo ECG (Hz)", f"{bio_base}.sampling_rate"),
                    ("Señales (ecg,hr,hrv)", f"{bio_base}.signals"),
                ],
            },
            {
                "title": "Robustez Adversarial", 
                "items": [
                    ("Niveles de ruido", f"{adv_base}.noise_levels"),
                    ("Escala máxima de ruido", f"{adv_base}.max_noise_scale"),
                ],
            },
            {
                "title": "Seguridad (Bio-Gating)", 
                "items": [
                    ("Activado", f"{saf_base}.bio_gating"),
                    ("Umbral de estrés", f"{saf_base}.stress_threshold"),
                    ("Modo de activación", f"{saf_base}.activation_mode"),
                ],
            },
            {
                "title": "Visualización", 
                "items": [
                    ("Tema (dark/light)", "visualization.theme"),
                    ("Puerto del servidor", "visualization.server_port"),
                    ("DPI", "visualization.dpi"),
                    ("Formato", "visualization.format"),
                    ("Interactividad", "visualization.interactive"),
                ],
            },
            {
                "title": "Despliegue", 
                "items": [
                    ("Destino", "deployment.target"),
                    ("Cuantización", "deployment.quantization"),
                    ("Timeout (s)", "deployment.timeout"),
                    ("Monitoreo", "deployment.monitoring"),
                    ("Auto-rollback", "deployment.auto_rollback"),
                ],
            },
        ]

        print("Config rápida y guiada. Selecciona una sección para editar:")
        for i, sec in enumerate(menu, start=1):
            print(f"  {i}. {sec['title']}")
        print("  0. Salir")

        try:
            choice = input("\nSección (0-7): ").strip()
        except EOFError:
            print(f"\n{Colors.YELLOW}⚠️ Entrada no disponible; saliendo de configuración.{Colors.ENDC}\n")
            return

        if choice in ("0", ""):
            print("\nSaliendo sin cambios.\n")
            return
        
        try:
            idx = int(choice) - 1
            section = menu[idx]
        except Exception:
            print(f"\n{Colors.RED}✗ Selección inválida.{Colors.ENDC}\n")
            return

        print(f"\n{Colors.BOLD}→ {section['title']}{Colors.ENDC}")
        print("(Pulsa Enter para mantener el valor actual)\n")

        # Editar cada item de la sección
        for label, path in section["items"]:
            try:
                current = get_path(self.config, path)
            except KeyError:
                # compat: aceptar 'training.*' si existe
                compat_path = path.replace(rl_base, "training")
                try:
                    current = get_path(self.config, compat_path)
                    path = compat_path
                except KeyError:
                    # si no existe, crear con valor por defecto razonable
                    current = ""
                    set_path(self.config, path, current)
            print(f"  • {label}: {Colors.CYAN}{current}{Colors.ENDC}")
            # Mostrar ayuda contextual
            hint = help_texts.get(path)
            if hint:
                print(f"    Ayuda: {hint}")
            else:
                # Ayuda genérica por tipo
                if isinstance(current, bool):
                    print("    Ayuda: escribe 'sí' o 'no' para activar o desactivar.")
                elif isinstance(current, (int, float)):
                    print("    Ayuda: escribe un número. Ejemplo: 10, 0.0003")
                elif isinstance(current, list):
                    print("    Ayuda: escribe una lista separada por comas. Ejemplo: ecg, hr, hrv")
                else:
                    print("    Ayuda: escribe texto. Ejemplo: PPO, dark, local")
            new_val = input("    Nuevo valor: ")
            try:
                casted = cast_value(current, new_val)
                set_path(self.config, path, casted)
            except Exception:
                print(f"    {Colors.YELLOW}⚠️ No se pudo actualizar '{label}', se mantiene el valor.{Colors.ENDC}")

        self.save_config()
        print(f"\n{Colors.GREEN}✓ Configuración actualizada y guardada{Colors.ENDC}\n")
    
    def run_demos(self):
        """Ejecutar todas las demostraciones"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}[DEMOSTRACIONES]{Colors.ENDC}\n")
        
        cmd = ["python3", str(self.root_dir / "INTERACTIVE_DEMOS.py")]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"\n{Colors.GREEN}✓ Todas las demos completadas{Colors.ENDC}\n")
        except subprocess.CalledProcessError as e:
            print(f"\n{Colors.RED}✗ Error en demos: {e}{Colors.ENDC}\n")
    
    def documentation(self):
        """Mostrar documentación"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}[DOCUMENTACIÓN]{Colors.ENDC}\n")
        
        docs = {
            "1": ("COMPLETE_SYSTEM_INDEX.md", "📍 Índice central del sistema"),
            "2": ("DETAILED_ANALYSIS_REPORT.md", "📊 Análisis técnico detallado"),
            "3": ("CUSTOMIZATION_GUIDE.md", "⚙️ Guía de personalización"),
            "4": ("PRODUCTION_DEPLOYMENT_PLAN.md", "🚀 Plan de despliegue"),
            "5": ("EXECUTIVE_SUMMARY_FINAL.md", "📈 Resumen ejecutivo"),
        }
        
        for key, (file, desc) in docs.items():
            print(f"  {key}. {desc}")
            print(f"     📄 {file}")
        
        print()
        choice = input("Selecciona documento (1-5) o 0 para atrás: ").strip()
        
        if choice in docs:
            doc_file = self.root_dir / docs[choice][0]
            if doc_file.exists():
                subprocess.run(["less", str(doc_file)])
        
        print()
    
    def run(self):
        """Bucle principal interactivo"""
        self.print_banner()
        
        while True:
            self.print_menu()
            choice = input(f"{Colors.BOLD}Selecciona opción (0-7): {Colors.ENDC}").strip()
            
            if choice == "1":
                self.train()
            elif choice == "2":
                self.deploy()
            elif choice == "3":
                self.analyze()
            elif choice == "4":
                self.visualize()
            elif choice == "5":
                self.configure()
            elif choice == "6":
                self.run_demos()
            elif choice == "7":
                self.documentation()
            elif choice == "0":
                print(f"\n{Colors.GREEN}¡Hasta luego! 🏍️{Colors.ENDC}\n")
                sys.exit(0)
            else:
                print(f"{Colors.RED}Opción inválida{Colors.ENDC}\n")


def main():
    """Punto de entrada principal"""
    parser = argparse.ArgumentParser(
        description="Sistema de Coaching Bio-Adaptativo - CLI Central",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python3 system_cli.py                    # Modo interactivo
  python3 system_cli.py train --episodes 10
  python3 system_cli.py deploy --target production
  python3 system_cli.py analyze
  python3 system_cli.py visualize
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Entrenar modelo")
    train_parser.add_argument("--episodes", type=int, help="Número de episodios")
    train_parser.add_argument("--algorithm", help="Algoritmo RL")
    
    # Deploy
    deploy_parser = subparsers.add_parser("deploy", help="Desplegar sistema")
    deploy_parser.add_argument("--target", help="Destino (local/staging/production)")
    
    # Analyze
    subparsers.add_parser("analyze", help="Analizar resultados")
    
    # Visualize
    visualize_parser = subparsers.add_parser("visualize", help="Abrir dashboard o UI web")
    visualize_parser.add_argument("--mode", choices=["html", "gradio"], help="Selecciona modo de visualización")
    
    # Config
    subparsers.add_parser("configure", help="Configurar parámetros")
    
    # Demos
    subparsers.add_parser("demos", help="Ejecutar todas las demostraciones")
    
    # Docs
    subparsers.add_parser("docs", help="Ver documentación")
    
    args = parser.parse_args()
    
    manager = SystemManager()
    
    if args.command == "train":
        manager.train(
            episodes=args.episodes,
            algorithm=args.algorithm
        )
    elif args.command == "deploy":
        manager.deploy(target=args.target)
    elif args.command == "analyze":
        manager.analyze()
    elif args.command == "visualize":
        manager.visualize(mode=getattr(args, "mode", None))
    elif args.command == "configure":
        manager.configure()
    elif args.command == "demos":
        manager.run_demos()
    elif args.command == "docs":
        manager.documentation()
    else:
        # Modo interactivo
        manager.run()


if __name__ == "__main__":
    main()
