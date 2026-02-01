#!/usr/bin/env python3
"""
Gradio UI for Bio-Adaptive Coaching System
Provides an accessible web interface: presets, training, analysis, dashboard, and config.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

import gradio as gr

# Import SystemManager con ruta robusta
try:
    from system.core.system_cli import SystemManager
except ModuleNotFoundError:
    # Ajustar sys.path al root del proyecto
    ROOT = Path(__file__).resolve().parent.parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from system.core.system_cli import SystemManager


def format_metrics(data: Dict[str, Any]) -> str:
    lines = []
    if 'biometric' in data:
        b = data['biometric']
        lines += [
            "# Métricas Biométricas",
            f"- FC Media: {b.get('mean_hr', 0):.1f} bpm",
            f"- Variabilidad: {b.get('std_hr', 0):.1f} bpm",
            f"- RMSSD: {b.get('rmssd', 0):.4f}",
            f"- Estrés: {b.get('stress_level', 0):.1f}%",
            "",
        ]
    if 'training' in data:
        t = data['training']
        lines += [
            "# Métricas de Entrenamiento (RL)",
        ]
        if 'episodes' in t:
            lines += [
                f"- Episodios: {t.get('episodes', 0)}",
                f"- Recompensa Media: {t.get('mean_reward', 0):.2f}",
                f"- Recompensa Máx: {t.get('max_reward', 0):.2f}",
                "",
            ]
        elif 'episode_rewards' in t:
            rewards = t['episode_rewards']
            if rewards:
                lines += [
                    f"- Episodios ejecutados: {len(rewards)}",
                    f"- Recompensa Media: {sum(rewards)/len(rewards):.2f}",
                    f"- Recompensa Máx: {max(rewards):.2f}",
                    f"- Recompensa Mín: {min(rewards):.2f}",
                    "",
                ]
    if 'simulation' in data:
        s = data['simulation']
        lines += [
            "# Métricas de Simulación",
            f"- Velocidad Max: {s.get('max_velocity', 0):.1f} km/h",
            f"- Inclinación: {s.get('max_lean_angle', 0):.1f}°",
            f"- Aceleración: {s.get('mean_acceleration', 0):.2f} m/s²",
            "",
        ]
    if 'adversarial' in data:
        a = data['adversarial']
        lines += [
            "# Robustez Adversarial",
            f"- Mejora: +{a.get('mean_improvement', 0):.2f}%",
            f"- Robustez Max Ruido: {a.get('robustness_at_max_noise', 0):.2f}%",
            "",
        ]
    return "\n".join(lines) or "(Sin métricas)"


def launch():
    mgr = SystemManager()
    root = mgr.root_dir
    port = mgr.config.get('visualization', {}).get('server_port', 7860)

    # Helpers
    def apply_preset_action(preset: str):
        msg = mgr.apply_preset(preset)
        mgr.save_config()
        return f"{msg}. Config guardada."

    def update_rl(episodes: int, lr: float, batch: int, gamma: float, theme: str):
        rl = mgr.config['components']['reinforcement_learning']
        rl['episodes'] = int(episodes)
        rl['learning_rate'] = float(lr)
        rl['batch_size'] = int(batch)
        rl['gamma'] = float(gamma)
        # Actualizar tema visualización
        vis = mgr.config.get('visualization', {})
        vis['theme'] = (theme or vis.get('theme') or 'dark')
        mgr.config['visualization'] = vis
        mgr.save_config()
        return "Parámetros de entrenamiento y tema actualizados."

    def quick_train(episodes: int):
        # Small training using moto_bio_project
        try:
            moto_src = root / 'moto_bio_project' / 'src'
            sys.path.insert(0, str(moto_src))
            from train import create_training_environment, train_ppo_agent
            env, df = create_training_environment(n_laps=max(1, episodes // 2))
            model, metrics = train_ppo_agent(env, total_timesteps=max(200, episodes * 200))
            return json.dumps(metrics, indent=2)
        except Exception as e:
            return f"Error en entrenamiento: {e}"

    def analyze_action():
        candidates = [
            root / 'DEPLOYMENT_ARTIFACTS' / 'demo_results.json',
            root / 'workspace' / 'results' / 'demo_results.json',
            root / 'demo_results.json',
        ]
        for p in candidates:
            if p.exists():
                data = json.loads(p.read_text())
                return format_metrics(data)
        return "No se encontraron resultados. Ejecuta 'Demos' o 'Entrenamiento' primero."

    def show_dashboard():
        dashboard = root / 'dashboard.html'
        if dashboard.exists():
            return dashboard.read_text(encoding='utf-8')
        return "<h3>Dashboard no encontrado</h3>"

    def run_deploy():
        try:
            out = subprocess.run(['python3', str(root / 'main.py'), 'deploy'], check=False, capture_output=True, text=True)
            return out.stdout or out.stderr
        except Exception as e:
            return f"Error en despliegue: {e}"

    # Presets extra solicitados: demo y científico
    def apply_demo():
        msg = mgr.apply_preset('fast')
        vis = mgr.config['visualization']
        vis['theme'] = 'dark'
        vis['server_port'] = port
        mgr.save_config()
        return "Preset demo aplicado (rápido, tema oscuro)."

    def apply_science():
        msg = mgr.apply_preset('robust')
        rl = mgr.config['components']['reinforcement_learning']
        rl['episodes'] = 100
        rl['learning_rate'] = 0.0001
        rl['batch_size'] = 256
        rl['gamma'] = 0.999
        adv = mgr.config['components']['adversarial_training']
        adv['noise_levels'] = 150
        sim = mgr.config['components']['simulation']
        sim['timesteps'] = 1000
        mgr.save_config()
        return "Preset exploración científica aplicado (entrenamiento profundo y robustez)."

    with gr.Blocks(title="Moto Bio-Adaptive Coaching UI") as demo:
        gr.Markdown("""
        # 🏍️ Bio-Adaptive Coaching UI
        Interfaz accesible para configurar, entrenar, analizar y visualizar.
        """)

        with gr.Tab("Presets"):
            gr.Markdown("Selecciona un preset para configurar el sistema rápidamente.")
            with gr.Row():
                btn_fast = gr.Button("Aplicar Entrenamiento rápido")
                btn_robust = gr.Button("Aplicar Entrenamiento robusto")
            with gr.Row():
                btn_demo = gr.Button("Aplicar Presentación demo")
                btn_science = gr.Button("Aplicar Exploración científica")
            out_preset = gr.Textbox(label="Resultado")
            btn_fast.click(lambda: apply_preset_action('fast'), outputs=out_preset)
            btn_robust.click(lambda: apply_preset_action('robust'), outputs=out_preset)
            btn_demo.click(apply_demo, outputs=out_preset)
            btn_science.click(apply_science, outputs=out_preset)

        with gr.Tab("Entrenamiento"):
            gr.Markdown("Entrenamiento rápido (demostración). Para sesiones largas usa CLI.")
            episodes = gr.Slider(1, 100, value=5, step=1, label="Episodios (demo)")
            btn_train = gr.Button("Entrenar (rápido)")
            train_out = gr.Textbox(label="Resultados del entrenamiento")
            btn_train.click(quick_train, inputs=episodes, outputs=train_out)

        with gr.Tab("Análisis"):
            gr.Markdown("Analiza y resume métricas de resultados.")
            btn_an = gr.Button("Analizar")
            an_out = gr.Markdown()
            btn_an.click(analyze_action, outputs=an_out)

        with gr.Tab("Dashboard"):
            gr.Markdown("Visualización integrada del dashboard HTML.")
            html = gr.HTML()
            btn_refresh = gr.Button("Refrescar")
            btn_refresh.click(show_dashboard, outputs=html)

        with gr.Tab("Configurar (RL)"):
            gr.Markdown("Edita parámetros principales de entrenamiento.")
            rl = mgr.config['components']['reinforcement_learning']
            vis_cfg = mgr.config.get('visualization', {})
            inp_ep = gr.Slider(1, 200, value=rl.get('episodes', 5), step=1, label="Episodios")
            inp_lr = gr.Number(value=rl.get('learning_rate', 0.0003), label="Tasa de aprendizaje")
            inp_bs = gr.Slider(16, 512, value=rl.get('batch_size', 64), step=16, label="Tamaño de batch")
            inp_gm = gr.Number(value=rl.get('gamma', 0.99), label="Gamma")
            theme_sel = gr.Radio(choices=["dark", "light"], value=vis_cfg.get('theme', 'dark'), label="Tema (UI)")
            btn_save = gr.Button("Guardar cambios")
            save_out = gr.Textbox(label="Estado")
            btn_save.click(update_rl, inputs=[inp_ep, inp_lr, inp_bs, inp_gm, theme_sel], outputs=save_out)

        with gr.Tab("Despliegue"):
            gr.Markdown("Ejecuta despliegue y muestra salida.")
            btn_dep = gr.Button("Desplegar")
            dep_out = gr.Textbox(lines=20, label="Log de despliegue")
            btn_dep.click(run_deploy, outputs=dep_out)

    # Lanzar servidor
    demo.launch(server_name="0.0.0.0", server_port=int(port), show_error=True)


if __name__ == "__main__":
    launch()
