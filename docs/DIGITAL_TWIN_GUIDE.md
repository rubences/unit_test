# Digital Twin Visualizer - Guía de Integración

## 📋 Índice
1. [Descripción General](#descripción-general)
2. [Arquitectura](#arquitectura)
3. [Opción A: Three.js (Web)](#opción-a-threejs-web)
4. [Opción B: Unity C#](#opción-b-unity-c)
5. [Integración con Pipeline de Entrenamiento](#integración-con-pipeline-de-entrenamiento)
6. [Troubleshooting](#troubleshooting)

---

## Descripción General

El **Digital Twin Visualizer** es un sistema de visualización 3D en tiempo real que:

✅ **Conecta** al servidor WebSocket de Python (`socket_bridge.py`)  
✅ **Visualiza** la posición y rotación de la motocicleta en 3D  
✅ **Dibuja** trayectorias: REAL (verde) vs PREDICCIÓN (roja)  
✅ **Muestra** métricas en tiempo real (speed, throttle, brake, reward)  
✅ **Soporta** dos plataformas:
   - **Opción A**: Three.js (navegador web, sin dependencias)
   - **Opción B**: Unity C# (motor gráfico profesional)

---

## Arquitectura

```
┌─────────────────────────────────────────────┐
│     Entrenamiento RL (Python)               │
│  ├─ Gymnasium Env                           │
│  ├─ RL Agent (PPO/A2C/DQN)                  │
│  └─ Adversarial Training                    │
└─────────────┬───────────────────────────────┘
              │ env.step()
              ▼
┌─────────────────────────────────────────────┐
│  Socket Bridge Server (socket_bridge.py)    │
│  ├─ EnvironmentBridge                       │
│  ├─ SocketBridgeServer (WebSocket)          │
│  └─ MotorcycleTelemetry (JSON payloads)     │
└─────────────┬───────────────────────────────┘
              │ JSON via WebSocket :5555
              ▼
     ┌────────────────────┐
     │   VISUALIZADOR 3D  │
     ├────────────────────┤
     │ Opción A: Three.js │  ← Recomendado para desarrollo
     │ Opción B: Unity    │  ← Recomendado para profesional
     └────────────────────┘
```

### Flujo de Datos

```json
// Python → WebSocket
{
  "type": "telemetry",
  "data": {
    "position": [x, y, z],           // Posición en metros
    "rotation": [roll, pitch, yaw],   // Rotación en radianes
    "velocity": [vx, vy, vz],         // Velocidad en m/s
    "speed": 25.5,                    // Velocidad escalar
    "throttle": 0.75,                 // 0-1
    "brake": 0.0,                     // 0-1
    "lean_angle": 5.2,                // Grados
    "prediction": [x_pred, y_pred, z_pred],  // Predicción AI
    "reward": 1.5,                    // Recompensa
    "episode_info": {
      "step": 125,
      "episode": 3,
      "done": false
    }
  }
}
```

---

## Opción A: Three.js (Web)

### 🚀 Quick Start

#### 1. Iniciar Servidor Python

```bash
cd /workspaces/Coaching-for-Competitive-Motorcycle-Racing

# En terminal 1: Servidor WebSocket
python src/deployment/socket_bridge.py

# En terminal 2: Ejemplo de demostración
python -c "
from src.deployment.socket_bridge import example_demo
import asyncio
asyncio.run(example_demo())
"
```

**Esperado**:
```
Server listening on 0.0.0.0:5555
Broadcasting telemetry...
Client connected
Sent telemetry packet...
```

#### 2. Abrir Visualizador

```bash
# Opción A: Abrir directamente desde VS Code
"$BROWSER" file:///workspaces/Coaching-for-Competitive-Motorcycle-Racing/src/visualization/motorcycle_visualizer.html

# Opción B: Usar servidor HTTP simple
cd src/visualization
python -m http.server 8000
# Luego abrir: http://localhost:8000/motorcycle_visualizer.html
```

#### 3. Ver Visualización

- ✅ **Moto azul** en el centro
- ✅ **Línea verde** = trayectoria real
- ✅ **Línea roja** = trayectoria predicha
- ✅ **HUD izquierdo** = posición, rotación, controles
- ✅ **HUD derecho** = estadísticas de trayectorias
- ✅ **Leyenda abajo izquierda** = referencia de colores

### 🎮 Controles

| Tecla | Acción |
|-------|--------|
| `R` | Reset trayectorias |
| Mouse Rueda | Zoom (automático con cámara) |
| Cámara | Sigue automáticamente a la moto |

### 📊 Interfaz HUD

**Panel Izquierdo**:
```
📍 POSICIÓN
X: 12.345
Y:  0.500
Z: -8.234

🎯 ROTACIÓN
Roll:  0.000°
Pitch: 0.000°
Yaw:  45.123°

⚡ CONTROL
Speed: 25.50 m/s
Throttle: 75%
Brake: 0%
```

**Panel Derecho**:
```
📈 TRAYECTORIAS
Real:        234 pts
Predicción:  234 pts
Error:       1.23 m
```

**Panel Inferior Izquierda** (Leyenda):
```
🎨 LEYENDA
🔴 Trayectoria Predicha
🟢 Trayectoria Real
🔵 Moto 3D
⚫ Pista
```

### 🔧 Personalización

#### Cambiar URL del Servidor

Editar `motorcycle_visualizer.html` línea ~600:

```javascript
const serverUrl = 'ws://localhost:5555';  // ← Cambiar aquí
```

#### Cambiar Colores de Trayectorias

```javascript
// Línea ~330 (trayectoria real - verde)
const materialReal = new THREE.LineBasicMaterial({ color: 0x00ff00 });

// Línea ~338 (trayectoria predicha - rojo)
const materialPredicted = new THREE.LineBasicMaterial({ color: 0xff0000 });
```

#### Ajustar Máximo de Puntos

```javascript
// Línea ~555 (limitar buffer para performance)
const maxTrajectoryPoints = 500;  // ← Cambiar a 1000 para más historia
```

#### Cambiar Modelo 3D de la Moto

Reemplazar `createMotorcycle()` con importar modelo glTF:

```javascript
function createMotorcycle() {
    const loader = new THREE.GLTFLoader();
    loader.load('models/motorcycle.gltf', (gltf) => {
        motorcycle = gltf.scene;
        scene.add(motorcycle);
    });
}
```

### 🎨 Estilos CSS Avanzados

Personalizar tema oscuro/claro:

```css
body {
    background: #ffffff;  /* Cambiar de negro a blanco */
    color: #000000;
}

#hud {
    background: rgba(255, 255, 255, 0.9);
    border: 2px solid #0000ff;  /* Cambiar color borde */
}
```

### 📱 Responsive Design

Three.js automáticamente se adapta a cambios de ventana:
- ✅ Funciona en tablets
- ✅ Funciona en phones (orientación horizontal recomendada)
- ⚠️ Performance reducido en dispositivos móviles

### ⚡ Performance

**Optimizaciones aplicadas**:
- ✅ FOG: limita renderizado lejano
- ✅ Buffer circular: máx 500 puntos
- ✅ Geometry pooling: reutiliza buffers
- ✅ Shadow maps: activadas

**FPS esperado**:
- 60 FPS: Desktop moderno
- 30 FPS: Laptop estándar
- 15 FPS: Mobile/tablet

---

## Opción B: Unity C#

### 🚀 Quick Start

#### 1. Setup Proyecto Unity

```bash
# 1. Crear o abrir proyecto Unity (2021.3+)
# 2. Importar WebSocketSharp (NuGet):
#    Assets > Import Package > Custom Package
#    O usar Package Manager → Add package from git URL
```

#### 2. Instalar Dependencias

En Package Manager de Unity:

```
WebSocketSharp: https://github.com/sta/WebSocketSharp.git
Newtonsoft.Json: com.unity.nuget.newtonsoft-json
```

#### 3. Configurar Scene

1. **Crear GameObject vacío**: `DigitalTwinManager`
   
2. **Adjuntar script**: `TelemetryReceiver.cs`
   ```csharp
   // En Inspector:
   - Server URL: ws://localhost:5555
   - Reconnect Delay: 3
   - Show Debug Info: true
   ```

3. **Crear Prefab Motocicleta** (opcional):
   - Crear model de moto con geometría
   - Guardar como prefab
   - Asignar en `motorcyclePrefab`

4. **Crear Line Renderers**:
   ```csharp
   // Real Trajectory (verde)
   GameObject lineReal = new GameObject("RealTrajectory");
   LineRenderer lr = lineReal.AddComponent<LineRenderer>();
   
   // Predicted Trajectory (rojo)
   GameObject linePred = new GameObject("PredictedTrajectory");
   LineRenderer lr = linePred.AddComponent<LineRenderer>();
   ```

#### 4. Ejecutar

```bash
# Terminal 1: Servidor WebSocket
python src/deployment/socket_bridge.py

# Terminal 2: Entrenamiento
python src/training/adversarial_training.py

# Unity: Press Play ▶️
```

### 🎮 Controles

| Tecla | Acción |
|-------|--------|
| `R` | Reset trayectorias |
| `F1` | Mostrar estadísticas en consola |

### 📊 Integración con Script

```csharp
// Acceder desde otro script
TelemetryReceiver receiver = GetComponent<TelemetryReceiver>();

// Verificar conexión
if (receiver.IsConnected) {
    Debug.Log("Connected!");
}

// Obtener datos de trayectoria
int realPoints = receiver.RealTrajectoryPointCount;
int predPoints = receiver.PredictedTrajectoryPointCount;
```

### 🔧 Personalización

#### Cambiar Servidor

```csharp
// En Inspector o en código:
[SerializeField] private string serverUrl = "ws://tu_servidor:5555";
```

#### Cambiar Modelo de Moto

```csharp
// En OnGUI o Inspector:
GameObject myMotorcycle = Resources.Load<GameObject>("Models/MyMotorcycle");
telemetryReceiver.SetMotorcyclePrefab(myMotorcycle);
```

#### Filtrar Actualizaciones

```csharp
// En HandleWebSocketMessage():
// Procesar solo cada N frames
if (frameCount++ % 3 == 0) {  // Cada 3er frame
    UpdateMotorcycleTransform(telemetry);
}
```

### 🎥 Implementar Cámara Custom

```csharp
void LateUpdate() {
    if (motorcycleTransform == null) return;
    
    // Chase camera (siguiendo a moto)
    Vector3 offset = motorcycleTransform.forward * -5f + Vector3.up * 3f;
    Camera.main.transform.position = motorcycleTransform.position + offset;
    Camera.main.transform.LookAt(motorcycleTransform.position + Vector3.up);
}
```

### 📈 Agregar UI Dashboard

```csharp
using UnityEngine.UI;

public class MotorcycleDashboard : MonoBehaviour {
    public Text speedText;
    public Text throttleText;
    public Slider speedSlider;
    public Image connectionStatus;
    
    public void UpdateDashboard(MotorcycleTelemetry telemetry) {
        speedText.text = $"Speed: {telemetry.speed:F1} m/s";
        throttleText.text = $"Throttle: {telemetry.throttle * 100:F0}%";
        speedSlider.value = telemetry.speed;
        
        connectionStatus.color = isConnected ? Color.green : Color.red;
    }
}
```

### 🎨 Shaders Personalizados

Renderizar trayectorias con efecto fade:

```glsl
Shader "Custom/TrajectoryFade" {
    SubShader {
        Pass {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            
            float4 vert(float4 pos : POSITION) : SV_POSITION {
                return UnityObjectToClipPos(pos);
            }
            
            float4 frag() : SV_Target {
                // Fade basado en tiempo
                return float4(1, 0, 0, sin(_Time.y) * 0.5 + 0.5);
            }
            ENDCG
        }
    }
}
```

---

## Integración con Pipeline de Entrenamiento

### Opción A: Ejecutar Simultáneamente

#### Terminal 1: Servidor WebSocket
```bash
python src/deployment/socket_bridge.py
```

#### Terminal 2: Entrenamiento
```bash
python src/training/adversarial_training.py
```

#### Terminal 3: Visualizador
```bash
# Three.js
"$BROWSER" file:///path/to/motorcycle_visualizer.html

# O Unity: Presionar Play
```

### Opción B: Script de Automatización

Crear `scripts/launch_digital_twin.py`:

```python
import subprocess
import time

# Iniciar servidor
server = subprocess.Popen(['python', 'src/deployment/socket_bridge.py'])
time.sleep(2)  # Esperar conexión

# Iniciar entrenamiento
training = subprocess.Popen(['python', 'src/training/adversarial_training.py'])

# Abrir visualizador
import os
os.system('open src/visualization/motorcycle_visualizer.html')

# Esperar a completación
training.wait()
server.terminate()
```

Ejecutar:
```bash
python scripts/launch_digital_twin.py
```

### Opción C: Notebook Jupyter

```python
# En Jupyter notebook
import subprocess
import time
from IPython.display import HTML

# Iniciar servidor en background
server = subprocess.Popen(['python', 'src/deployment/socket_bridge.py'])
time.sleep(2)

# Ejecutar entrenamiento
from src.training.adversarial_training import train_adversarial

results = train_adversarial(
    curriculum_enabled=True,
    total_timesteps=10000
)

# Mostrar iframe con visualizador
HTML('''
<iframe src="src/visualization/motorcycle_visualizer.html" 
        width="100%" height="800"></iframe>
''')

server.terminate()
```

---

## Troubleshooting

### ❌ "Desconectado" en visualizador

**Causa**: Servidor no está escuchando

**Solución**:
```bash
# 1. Verificar que socket_bridge.py esté corriendo
ps aux | grep socket_bridge

# 2. Probar conexión manual
python -c "
import asyncio
from websockets import connect

async def test():
    async with connect('ws://localhost:5555') as ws:
        print('✓ Conectado')

asyncio.run(test())
"

# 3. Si puerto ocupado, cambiar en socket_bridge.py:
# PORT = 5556
```

### ❌ "WebSocket error: connection refused"

**Causa**: Puerto bloqueado o firewall

**Solución**:
```bash
# Limpiar puerto
lsof -i :5555
kill -9 <PID>

# O cambiar puerto en code
```

### ❌ Moto no se mueve

**Causa**: Datos no llegando o malformados

**Solución**:
```javascript
// En console (F12) de navegador:
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

### ❌ Performance bajo (FPS bajo)

**Causa**: Buffer de trayectoria muy grande

**Solución**:
```javascript
// Reducir puntos
const maxTrajectoryPoints = 200;  // De 500

// O renderizar cada N puntos
if (trajectoryRealPoints.length % 2 === 0) {
    updateTrajectories();
}
```

### ❌ Líneas de trayectoria no visibles en Unity

**Causa**: Material no configurado

**Solución**:
```csharp
void ConfigureLineRenderer(LineRenderer lr, Color color) {
    lr.material = new Material(Shader.Find("Sprites/Default"));
    lr.startColor = color;
    lr.endColor = color;
    lr.startWidth = 0.2f;
    lr.endWidth = 0.2f;
    lr.sortingOrder = 10;  // Renderizar arriba
}
```

### ❌ "JSON.parse error" en Three.js

**Causa**: Mensaje malformado del servidor

**Solución**:
```javascript
ws.onmessage = (event) => {
    try {
        const message = JSON.parse(event.data);
        // OK
    } catch (e) {
        console.error('Mensaje no es JSON:', event.data);
    }
};
```

---

## 📚 Referencias Rápidas

### Archivos Clave

| Archivo | Propósito |
|---------|-----------|
| `src/deployment/socket_bridge.py` | Servidor WebSocket |
| `src/visualization/motorcycle_visualizer.html` | Cliente Three.js |
| `src/visualization/unity/TelemetryReceiver.cs` | Script Unity |
| `configs/train_config.yaml` | Config entrenamiento |

### Puertos Comunes

| Servicio | Puerto |
|----------|--------|
| WebSocket (Socket Bridge) | 5555 |
| HTTP (servidor archivos) | 8000 |
| Unity Editor | 5037 |

### Librerías Externas

**Three.js**:
- CDN: `https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js`
- Documentación: `https://threejs.org/docs`

**WebSocketSharp** (Unity):
- GitHub: `https://github.com/sta/WebSocketSharp`
- Docs: Comentarios en código

---

## 🎓 Casos de Uso

### Caso 1: Visualizar Entrenamiento en Vivo

```bash
# Terminal 1
python src/deployment/socket_bridge.py

# Terminal 2
python src/training/adversarial_training.py --visualize

# Terminal 3
open src/visualization/motorcycle_visualizer.html
```

### Caso 2: Depurar Predicciones del Modelo

```python
# Visualizar predicción vs realidad
# Las líneas roja (predicción) y verde (real) mostrarán divergencia
# si el modelo está teniendo errores de predicción
```

### Caso 3: Analizar Comportamiento en Pista

```
# Ver cómo la moto recorre la pista
# Observar inclinaciones (lean angle)
# Analizar aceleración/frenado (throttle/brake)
```

---

## 📞 Soporte

Para problemas:

1. **Revisar logs**:
   ```bash
   cat socket_bridge.log
   ```

2. **Activar debug**:
   ```python
   # socket_bridge.py
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Test unitario**:
   ```bash
   pytest tests/test_socket_bridge.py -v
   ```

---

**Última actualización**: 2024-12-19  
**Versión**: 1.0  
**Estado**: ✅ Producción
