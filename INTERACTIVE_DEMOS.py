#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     🏍️ INTERACTIVE DEMOS - Sistema Coaching Adaptativo Háptico          ║
║                                                                           ║
║  Este script ejecuta demostraciones interactivas de todos los            ║
║  componentes principales del sistema                                     ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Styling
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Imprimir encabezado con estilos"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}\n")

def print_success(text: str):
    """Imprimir mensaje de éxito"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")

def print_info(text: str):
    """Imprimir mensaje informativo"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")

def print_warning(text: str):
    """Imprimir advertencia"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")

class BiometricDemo:
    """Demo: Sistema Biométrico"""
    
    @staticmethod
    def run():
        print_header("🏥 DEMO 1: SISTEMA BIOMÉTRICO - ECG & HRV")
        
        try:
            import neurokit2 as nk
            print_info("Librería NeuroKit2 disponible")
        except ImportError:
            print_warning("NeuroKit2 no disponible, usando datos sintéticos")
        
        # Generar datos ECG sintéticos
        print_info("Generando datos ECG sintéticos (10 segundos)...")
        sampling_rate = 250  # Hz
        duration = 10  # segundos
        t = np.arange(0, duration, 1/sampling_rate)
        
        # ECG sintético realista
        ecg_signal = (
            0.8 * np.sin(2 * np.pi * 1.2 * t) +  # P wave
            0.5 * np.cos(2 * np.pi * 5 * t) +     # QRS complex
            0.3 * np.sin(2 * np.pi * 0.8 * t) +   # T wave
            0.1 * np.random.randn(len(t))         # Ruido
        )
        
        # Calcular métricas HRV
        heart_rate = 60 + 20 * np.sin(2 * np.pi * 0.1 * t)  # Variación HR simulada
        mean_hr = np.mean(heart_rate)
        std_hr = np.std(heart_rate)
        rmssd = np.sqrt(np.mean(np.diff(heart_rate)**2))
        
        print_success(f"ECG generado: {len(ecg_signal)} samples")
        print_success(f"Frecuencia cardíaca media: {mean_hr:.1f} bpm")
        print_success(f"Desviación estándar: {std_hr:.1f} bpm")
        print_success(f"RMSSD (variabilidad): {rmssd:.2f} ms")
        
        # Detectar estrés
        stress_level = min(100, (abs(mean_hr - 70) / 50 + std_hr / 30) * 50)
        print_success(f"Nivel estimado de estrés: {stress_level:.1f}%")
        
        # Crear visualización
        fig, axes = plt.subplots(3, 1, figsize=(14, 8))
        
        # ECG
        axes[0].plot(t, ecg_signal, label='ECG Signal', linewidth=1.5)
        axes[0].set_ylabel('Amplitud (mV)')
        axes[0].set_title('Señal ECG Sintética', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Heart Rate
        axes[1].plot(t, heart_rate, label='Heart Rate', color='red', linewidth=1.5)
        axes[1].axhline(mean_hr, color='green', linestyle='--', label=f'Media: {mean_hr:.1f} bpm')
        axes[1].set_ylabel('Frecuencia Cardíaca (bpm)')
        axes[1].set_title('Variabilidad de Frecuencia Cardíaca', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Stress level over time
        stress_over_time = 30 + 20 * np.sin(2 * np.pi * 0.05 * t) + np.random.randn(len(t)) * 5
        axes[2].fill_between(t, stress_over_time, alpha=0.5, color='orange')
        axes[2].plot(t, stress_over_time, label='Stress Level', color='darkorange', linewidth=2)
        axes[2].set_xlabel('Tiempo (segundos)')
        axes[2].set_ylabel('Nivel de Estrés (%)')
        axes[2].set_title('Monitoreo de Estrés en Tiempo Real', fontweight='bold')
        axes[2].set_ylim([0, 100])
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = Path('DEPLOYMENT_ARTIFACTS/biometric_demo.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print_success(f"Visualización guardada: {output_path}")
        plt.close()
        
        return {
            'mean_hr': float(mean_hr),
            'std_hr': float(std_hr),
            'rmssd': float(rmssd),
            'stress_level': float(stress_level)
        }

class TrainingDemo:
    """Demo: Entrenamiento RL"""
    
    @staticmethod
    def run():
        print_header("🤖 DEMO 2: ENTRENAMIENTO POR REFUERZO")
        
        print_info("Simulando entrenamiento PPO por 500 pasos...")
        
        # Simular entrenamiento
        episodes = 5
        steps_per_episode = 100
        
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'learning_curve': [],
            'actor_loss': [],
            'critic_loss': []
        }
        
        best_reward = -np.inf
        
        for episode in range(episodes):
            # Simular episodio
            episode_reward = 0
            episode_length = 0
            
            for step in range(steps_per_episode):
                # Recompensa simulada
                reward = 10 * np.sin(step / 20) + np.random.randn() * 2
                episode_reward += reward
                episode_length += 1
                
                # Simular pérdidas
                actor_loss = 0.5 * np.exp(-episode / 5) + np.random.randn() * 0.01
                critic_loss = 0.3 * np.exp(-episode / 5) + np.random.randn() * 0.01
                
                metrics['actor_loss'].append(float(actor_loss))
                metrics['critic_loss'].append(float(critic_loss))
                metrics['learning_curve'].append(float(episode_reward / (step + 1)))
            
            metrics['episode_rewards'].append(float(episode_reward))
            metrics['episode_lengths'].append(int(episode_length))
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                print_success(f"Episodio {episode+1}: Recompensa={episode_reward:.2f} (✨ MEJOR)")
            else:
                print_info(f"Episodio {episode+1}: Recompensa={episode_reward:.2f}")
        
        # Visualizar entrenamiento
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Episode Rewards
        axes[0, 0].plot(metrics['episode_rewards'], marker='o', linewidth=2, markersize=8)
        axes[0, 0].set_title('Recompensas por Episodio', fontweight='bold')
        axes[0, 0].set_xlabel('Episodio')
        axes[0, 0].set_ylabel('Recompensa Acumulada')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning Curve
        axes[0, 1].plot(metrics['learning_curve'], linewidth=1.5, alpha=0.7)
        axes[0, 1].set_title('Curva de Aprendizaje', fontweight='bold')
        axes[0, 1].set_xlabel('Paso de Entrenamiento')
        axes[0, 1].set_ylabel('Recompensa Promedio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Actor Loss
        axes[1, 0].plot(metrics['actor_loss'], linewidth=1, alpha=0.7, color='red')
        axes[1, 0].set_title('Pérdida del Actor (Policy)', fontweight='bold')
        axes[1, 0].set_xlabel('Paso de Entrenamiento')
        axes[1, 0].set_ylabel('Pérdida')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Critic Loss
        axes[1, 1].plot(metrics['critic_loss'], linewidth=1, alpha=0.7, color='green')
        axes[1, 1].set_title('Pérdida del Crítico (Value)', fontweight='bold')
        axes[1, 1].set_xlabel('Paso de Entrenamiento')
        axes[1, 1].set_ylabel('Pérdida')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = Path('DEPLOYMENT_ARTIFACTS/training_demo.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print_success(f"Visualización guardada: {output_path}")
        plt.close()
        
        return metrics

class SimulationDemo:
    """Demo: Simulación de Motocicleta"""
    
    @staticmethod
    def run():
        print_header("🏍️ DEMO 3: SIMULACIÓN DE DINÁMICA DE MOTOCICLETA")
        
        print_info("Generando trayectoria de motocicleta en circuito simulado...")
        
        # Simular trayectoria
        time_steps = 300
        t = np.arange(time_steps) / 30  # 10 segundos a 30 Hz
        
        # Trayectoria en forma de 8 (infinity loop)
        x = 100 * np.sin(t) * (1 + 0.5 * np.cos(2*t))
        y = 50 * np.sin(2*t)
        
        # Velocidad
        dx = np.gradient(x)
        dy = np.gradient(y)
        velocity = np.sqrt(dx**2 + dy**2) * 30  # km/h
        
        # Ángulos de inclinación (lean angle)
        lean_angle = 30 * (1 + 0.8 * np.sin(2*np.pi*t/10))
        
        # Aceleración
        acceleration = np.gradient(velocity)
        
        print_success(f"Velocidad máxima: {np.max(velocity):.1f} km/h")
        print_success(f"Ángulo de inclinación máximo: {np.max(lean_angle):.1f}°")
        print_success(f"Aceleración media: {np.mean(np.abs(acceleration)):.2f} m/s²")
        
        # Crear visualización
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.35)
        
        # Trayectoria
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(x, y, linewidth=2, color='blue', label='Trayectoria')
        ax1.scatter(x[0], y[0], color='green', s=100, marker='o', label='Inicio', zorder=5)
        ax1.scatter(x[-1], y[-1], color='red', s=100, marker='x', label='Fin', zorder=5)
        ax1.set_xlabel('Eje X (m)')
        ax1.set_ylabel('Eje Y (m)')
        ax1.set_title('Trayectoria de Motocicleta en Circuito', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')
        
        # Velocidad
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(t, velocity, linewidth=2, color='orange')
        ax2.fill_between(t, velocity, alpha=0.3, color='orange')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Velocidad (km/h)')
        ax2.set_title('Perfil de Velocidad', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Lean Angle
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(t, lean_angle, linewidth=2, color='purple')
        ax3.fill_between(t, lean_angle, alpha=0.3, color='purple')
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Ángulo de Inclinación (°)')
        ax3.set_title('Ángulo de Inclinación (Lean Angle)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Aceleración
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(t, acceleration, linewidth=2, color='green')
        ax4.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax4.set_xlabel('Tiempo (s)')
        ax4.set_ylabel('Aceleración (m/s²)')
        ax4.set_title('Perfil de Aceleración', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Velocidad Angular
        heading = np.arctan2(dy, dx) * 180 / np.pi
        heading_rate = np.gradient(heading)
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(t, heading_rate, linewidth=2, color='red')
        ax5.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax5.set_xlabel('Tiempo (s)')
        ax5.set_ylabel('Velocidad Angular (°/s)')
        ax5.set_title('Velocidad Angular (Yaw Rate)', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Dinámica de Motocicleta - Simulación Completa', 
                     fontsize=14, fontweight='bold', y=0.995)
        
        output_path = Path('DEPLOYMENT_ARTIFACTS/simulation_demo.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print_success(f"Visualización guardada: {output_path}")
        plt.close()
        
        return {
            'max_velocity': float(np.max(velocity)),
            'max_lean_angle': float(np.max(lean_angle)),
            'mean_acceleration': float(np.mean(np.abs(acceleration))),
            'duration': float(t[-1])
        }

class AdversarialDemo:
    """Demo: Entrenamiento Adversarial"""
    
    @staticmethod
    def run():
        print_header("⚔️ DEMO 4: ENTRENAMIENTO ADVERSARIAL ROBUSTO")
        
        print_info("Entrenando agente robusto contra perturbaciones...")
        
        # Simular robustez contra perturbaciones
        noise_levels = np.linspace(0, 0.5, 50)
        agent_performance = []
        baseline_performance = []
        
        for noise in noise_levels:
            # Agente con adversarial training
            adv_perf = 100 * np.exp(-2*noise) * (0.8 + 0.2*np.random.random())
            
            # Baseline sin adversarial training
            baseline_perf = 100 * np.exp(-4*noise) * (0.7 + 0.3*np.random.random())
            
            agent_performance.append(float(adv_perf))
            baseline_performance.append(float(baseline_perf))
        
        improvement = np.array(agent_performance) - np.array(baseline_performance)
        mean_improvement = np.mean(improvement)
        
        print_success(f"Mejora media en robustez: {mean_improvement:.2f}%")
        print_success(f"Robustez en ruido máximo: {agent_performance[-1]:.1f}%")
        
        # Crear visualización
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Comparación de robustez
        axes[0].plot(noise_levels, agent_performance, marker='o', label='Con Adversarial Training', 
                    linewidth=2, markersize=5)
        axes[0].plot(noise_levels, baseline_performance, marker='s', label='Baseline (sin Adversarial)', 
                    linewidth=2, markersize=5, linestyle='--')
        axes[0].fill_between(noise_levels, agent_performance, baseline_performance, 
                            alpha=0.2, color='green')
        axes[0].set_xlabel('Nivel de Perturbación')
        axes[0].set_ylabel('Rendimiento (%)')
        axes[0].set_title('Robustez contra Perturbaciones', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Mejora
        axes[1].bar(range(0, len(improvement), 5), improvement[::5], color='green', alpha=0.7)
        axes[1].axhline(mean_improvement, color='red', linestyle='--', label=f'Media: {mean_improvement:.2f}%')
        axes[1].set_xlabel('Nivel de Perturbación')
        axes[1].set_ylabel('Mejora (%)')
        axes[1].set_title('Mejora de Robustez', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = Path('DEPLOYMENT_ARTIFACTS/adversarial_demo.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print_success(f"Visualización guardada: {output_path}")
        plt.close()
        
        return {
            'mean_improvement': float(mean_improvement),
            'robustness_at_max_noise': float(agent_performance[-1])
        }

class ComparisonDemo:
    """Demo: Comparación de Configuraciones"""
    
    @staticmethod
    def run():
        print_header("📊 DEMO 5: COMPARACIÓN DE CONFIGURACIONES")
        
        print_info("Comparando diferentes configuraciones del sistema...")
        
        # Configuraciones
        configs = {
            'Baseline': {
                'reward': 85,
                'robustness': 65,
                'latency': 150,
                'safety': 70
            },
            'Sin Bio-gating': {
                'reward': 92,
                'robustness': 45,
                'latency': 120,
                'safety': 30
            },
            'Con Bio-gating': {
                'reward': 88,
                'robustness': 85,
                'latency': 145,
                'safety': 95
            },
            'Optimizado': {
                'reward': 90,
                'robustness': 88,
                'latency': 140,
                'safety': 93
            }
        }
        
        for name, metrics in configs.items():
            print_info(f"{name}:")
            print(f"  • Recompensa: {metrics['reward']}%")
            print(f"  • Robustez: {metrics['robustness']}%")
            print(f"  • Latencia: {metrics['latency']}ms")
            print(f"  • Seguridad: {metrics['safety']}%")
        
        # Crear visualización
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3)
        
        # Radar chart
        from matplotlib.patches import Polygon
        ax1 = fig.add_subplot(gs[0, 0], projection='polar')
        
        categories = list(configs['Baseline'].keys())
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        colors_list = ['red', 'blue', 'green', 'orange']
        for (config_name, metrics), color in zip(configs.items(), colors_list):
            values = list(metrics.values())
            values += values[:1]
            ax1.plot(angles, values, 'o-', linewidth=2, label=config_name, color=color)
            ax1.fill(angles, values, alpha=0.15, color=color)
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 100)
        ax1.set_title('Comparación Multi-dimensional', fontweight='bold', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax1.grid(True)
        
        # Reward comparison
        ax2 = fig.add_subplot(gs[0, 1])
        rewards = [metrics['reward'] for metrics in configs.values()]
        ax2.bar(configs.keys(), rewards, color=colors_list, alpha=0.7)
        ax2.set_ylabel('Recompensa (%)')
        ax2.set_title('Comparación de Recompensas', fontweight='bold')
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(rewards):
            ax2.text(i, v+2, str(v), ha='center', fontweight='bold')
        
        # Safety comparison
        ax3 = fig.add_subplot(gs[1, 0])
        safety = [metrics['safety'] for metrics in configs.values()]
        ax3.bar(configs.keys(), safety, color=colors_list, alpha=0.7)
        ax3.set_ylabel('Seguridad (%)')
        ax3.set_title('Comparación de Seguridad', fontweight='bold')
        ax3.set_ylim([0, 100])
        ax3.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(safety):
            ax3.text(i, v+2, str(v), ha='center', fontweight='bold')
        
        # Latency comparison
        ax4 = fig.add_subplot(gs[1, 1])
        latency = [metrics['latency'] for metrics in configs.values()]
        ax4.bar(configs.keys(), latency, color=colors_list, alpha=0.7)
        ax4.set_ylabel('Latencia (ms)')
        ax4.set_title('Comparación de Latencia', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(latency):
            ax4.text(i, v+3, str(v), ha='center', fontweight='bold')
        
        plt.suptitle('Análisis Comparativo de Configuraciones', 
                     fontsize=14, fontweight='bold', y=0.995)
        
        output_path = Path('DEPLOYMENT_ARTIFACTS/comparison_demo.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print_success(f"Visualización guardada: {output_path}")
        plt.close()
        
        return configs

def main():
    """Ejecutar todos los demos"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                           ║")
    print("║          🏍️ INTERACTIVE DEMOS - Sistema Coaching Adaptativo 🏍️          ║")
    print("║                                                                           ║")
    print("║  Ejecución completa de demostraciones interactivas de todos los           ║")
    print("║  componentes del sistema                                                  ║")
    print("║                                                                           ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")
    
    start_time = datetime.now()
    results = {}
    
    try:
        # Demo 1: Biometría
        print_info("Iniciando Demo 1 de 5...")
        results['biometric'] = BiometricDemo.run()
        time.sleep(1)
        
        # Demo 2: Entrenamiento
        print_info("Iniciando Demo 2 de 5...")
        results['training'] = TrainingDemo.run()
        time.sleep(1)
        
        # Demo 3: Simulación
        print_info("Iniciando Demo 3 de 5...")
        results['simulation'] = SimulationDemo.run()
        time.sleep(1)
        
        # Demo 4: Adversarial
        print_info("Iniciando Demo 4 de 5...")
        results['adversarial'] = AdversarialDemo.run()
        time.sleep(1)
        
        # Demo 5: Comparación
        print_info("Iniciando Demo 5 de 5...")
        comparison = ComparisonDemo.run()
        results['comparison'] = {k: v for config_dict in comparison.values() 
                               for k, v in config_dict.items()}
        
        # Guardar resultados
        output_path = Path('DEPLOYMENT_ARTIFACTS/demo_results.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir a serializable
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        results_serializable = convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print_success(f"Resultados guardados: {output_path}")
        
        # Tiempo total
        duration = (datetime.now() - start_time).total_seconds()
        
        print_header("✅ RESUMEN DE DEMOS")
        print_success(f"Tiempo total: {duration:.1f} segundos")
        print_success("Todas las demostraciones completadas exitosamente")
        print_success("Visualizaciones guardadas en DEPLOYMENT_ARTIFACTS/")
        
        return True
        
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error durante la ejecución: {str(e)}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
