"""
Módulo de Visualización - Generador de gráficos e interfaz web
"""

from pathlib import Path
import subprocess

class VisualizationManager:
    """Gestor de visualizaciones"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def start_dashboard(self, port: int = 8080):
        """Iniciar dashboard interactivo"""
        dashboard = Path(__file__).parent.parent.parent / "dashboard.html"
        
        if not dashboard.exists():
            raise FileNotFoundError(f"Dashboard no encontrado: {dashboard}")
        
        # Verificar si servidor ya está corriendo
        try:
            import requests
            requests.head(f"http://localhost:{port}", timeout=1)
            print(f"✓ Servidor ya en ejecución en puerto {port}")
            return
        except:
            pass
        
        # Iniciar servidor
        print(f"Iniciando servidor en puerto {port}...")
        subprocess.Popen([
            "python3", "-m", "http.server", str(port),
            "--directory", str(dashboard.parent)
        ])
        
        print(f"✓ Dashboard disponible en: http://localhost:{port}/dashboard.html")
    
    def generate_biometric_chart(self, data: dict):
        """Generar gráfico biométrico"""
        pass
    
    def generate_training_chart(self, data: dict):
        """Generar gráfico de entrenamiento"""
        pass
    
    def generate_comparison_chart(self, data: dict):
        """Generar gráfico de comparación"""
        pass
