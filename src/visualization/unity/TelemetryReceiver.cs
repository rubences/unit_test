using UnityEngine;
using WebSocketSharp;
using Newtonsoft.Json;
using System.Collections.Generic;

/// <summary>
/// TelemetryReceiver.cs - Digital Twin receiver for motorcycle telemetry
/// Connects to Python WebSocket server and updates GameObject transforms
/// 
/// Usage:
/// 1. Attach to empty GameObject in scene
/// 2. Assign motorcycle prefab to motorcyclePrefab field
/// 3. Assign trail renderers for trajectory visualization
/// 4. Press Play - connects automatically
/// </summary>
public class TelemetryReceiver : MonoBehaviour
{
    [Header("WebSocket Configuration")]
    [SerializeField] private string serverUrl = "ws://localhost:5555";
    [SerializeField] private float reconnectDelay = 3f;

    [Header("Motorcycle Model")]
    [SerializeField] private GameObject motorcyclePrefab;
    [SerializeField] private Transform motorcycleTransform;

    [Header("Trajectory Visualization")]
    [SerializeField] private LineRenderer realTrajectory;
    [SerializeField] private LineRenderer predictedTrajectory;
    [SerializeField] private int maxTrajectoryPoints = 500;

    [Header("HUD Display")]
    [SerializeField] private Canvas hudCanvas;
    [SerializeField] private bool showDebugInfo = true;

    private WebSocket webSocket;
    private Queue<Vector3> realTrajectoryPoints = new Queue<Vector3>();
    private Queue<Vector3> predictedTrajectoryPoints = new Queue<Vector3>();
    private bool isConnected = false;
    private float lastUpdateTime = 0f;
    private int frameCount = 0;

    private struct MotorcycleTelemetry
    {
        public Vector3 position;
        public Vector3 rotation;
        public Vector3 velocity;
        public float speed;
        public float throttle;
        public float brake;
        public float leanAngle;
        public float reward;
        public EpisodeInfo episodeInfo;
    }

    private struct EpisodeInfo
    {
        public int step;
        public int episode;
        public bool done;
    }

    void Start()
    {
        // Create motorcycle instance if prefab assigned
        if (motorcyclePrefab != null && motorcycleTransform == null)
        {
            GameObject instance = Instantiate(motorcyclePrefab);
            motorcycleTransform = instance.transform;
        }

        // Initialize trajectory line renderers
        if (realTrajectory == null)
            realTrajectory = gameObject.AddComponent<LineRenderer>();
        
        if (predictedTrajectory == null)
            predictedTrajectory = gameObject.AddComponent<LineRenderer>();

        ConfigureLineRenderer(realTrajectory, new Color(0, 1, 0)); // Green for real
        ConfigureLineRenderer(predictedTrajectory, new Color(1, 0, 0)); // Red for predicted

        // Connect to WebSocket
        ConnectWebSocket();
    }

    void ConfigureLineRenderer(LineRenderer lineRenderer, Color color)
    {
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.startColor = color;
        lineRenderer.endColor = color;
        lineRenderer.startWidth = 0.1f;
        lineRenderer.endWidth = 0.1f;
    }

    void ConnectWebSocket()
    {
        if (webSocket != null)
            webSocket.Close();

        webSocket = new WebSocket(serverUrl);

        webSocket.OnOpen += () =>
        {
            isConnected = true;
            Debug.Log("✓ Conectado al servidor WebSocket");
        };

        webSocket.OnMessage += (sender, e) =>
        {
            HandleWebSocketMessage(e.Data);
        };

        webSocket.OnError += (sender, e) =>
        {
            Debug.LogError($"✗ WebSocket Error: {e.Message}");
        };

        webSocket.OnClose += (sender, e) =>
        {
            isConnected = false;
            Debug.Log("✗ Desconectado del servidor");
            Invoke("ConnectWebSocket", reconnectDelay);
        };

        webSocket.Connect();
    }

    void HandleWebSocketMessage(string data)
    {
        try
        {
            var jsonData = JsonConvert.DeserializeObject<dynamic>(data);

            if (jsonData["type"] == "telemetry")
            {
                var telemetryData = jsonData["data"];
                MotorcycleTelemetry telemetry = ParseTelemetry(telemetryData);
                
                UpdateMotorcycleTransform(telemetry);
                UpdateTrajectories(telemetry);
                UpdateHUD(telemetry);
            }
            else if (jsonData["type"] == "init")
            {
                // Initialize with historical trajectory
                var trajectory = jsonData["trajectory"];
                ClearTrajectories();
                // Load historical data if available
            }
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"Error parsing telemetry: {e.Message}");
        }
    }

    MotorcycleTelemetry ParseTelemetry(dynamic data)
    {
        var telemetry = new MotorcycleTelemetry
        {
            position = new Vector3(
                (float)data["position"][0],
                (float)data["position"][1],
                (float)data["position"][2]
            ),
            rotation = new Vector3(
                (float)data["rotation"][0],
                (float)data["rotation"][1],
                (float)data["rotation"][2]
            ),
            velocity = new Vector3(
                (float)data["velocity"][0],
                (float)data["velocity"][1],
                (float)data["velocity"][2]
            ),
            speed = (float)data["speed"],
            throttle = (float)data["throttle"],
            brake = (float)data["brake"],
            leanAngle = (float)data["lean_angle"],
            reward = (float)data["reward"],
            episodeInfo = new EpisodeInfo
            {
                step = (int)data["episode_info"]["step"],
                episode = (int)data["episode_info"]["episode"],
                done = (bool)data["episode_info"]["done"]
            }
        };

        return telemetry;
    }

    void UpdateMotorcycleTransform(MotorcycleTelemetry telemetry)
    {
        if (motorcycleTransform == null)
            return;

        // Update position (convert from Python coordinates if needed)
        motorcycleTransform.position = telemetry.position;

        // Update rotation (convert from radians to Euler angles)
        motorcycleTransform.eulerAngles = telemetry.rotation * Mathf.Rad2Deg;

        // Optional: Add tilt based on lean angle
        Vector3 tiltAxis = motorcycleTransform.forward;
        motorcycleTransform.RotateAround(motorcycleTransform.position, tiltAxis, telemetry.leanAngle);
    }

    void UpdateTrajectories(MotorcycleTelemetry telemetry)
    {
        // Add real position to trajectory
        realTrajectoryPoints.Enqueue(telemetry.position);
        if (realTrajectoryPoints.Count > maxTrajectoryPoints)
            realTrajectoryPoints.Dequeue();

        // Add predicted position to trajectory
        Vector3 predictedPos = new Vector3(
            telemetry.position.x + telemetry.velocity.x * 0.1f,
            telemetry.position.y + telemetry.velocity.y * 0.1f,
            telemetry.position.z + telemetry.velocity.z * 0.1f
        );
        predictedTrajectoryPoints.Enqueue(predictedPos);
        if (predictedTrajectoryPoints.Count > maxTrajectoryPoints)
            predictedTrajectoryPoints.Dequeue();

        // Update line renderers
        UpdateLineRenderer(realTrajectory, realTrajectoryPoints);
        UpdateLineRenderer(predictedTrajectory, predictedTrajectoryPoints);
    }

    void UpdateLineRenderer(LineRenderer lineRenderer, Queue<Vector3> points)
    {
        if (points.Count < 2)
            return;

        lineRenderer.positionCount = points.Count;
        int index = 0;
        foreach (Vector3 point in points)
        {
            lineRenderer.SetPosition(index, point);
            index++;
        }
    }

    void UpdateHUD(MotorcycleTelemetry telemetry)
    {
        if (!showDebugInfo)
            return;

        frameCount++;
        if (Time.realtimeSinceStartup - lastUpdateTime >= 0.1f) // Update every 0.1 seconds
        {
            OnGUI();
            lastUpdateTime = Time.realtimeSinceStartup;
        }
    }

    void OnGUI()
    {
        if (!showDebugInfo || !isConnected)
            return;

        GUILayout.BeginArea(new Rect(10, 10, 300, 400));
        GUILayout.Box("Motorcycle Digital Twin", GUILayout.ExpandWidth(true));

        GUILayout.Label($"Status: {(isConnected ? "CONNECTED ✓" : "DISCONNECTED ✗")}", 
            new GUIStyle(GUI.skin.label) { fontSize = 12, fontStyle = FontStyle.Bold });

        GUILayout.Space(10);

        GUILayout.Label("Position:", new GUIStyle(GUI.skin.label) { fontStyle = FontStyle.Bold });
        if (motorcycleTransform != null)
        {
            GUILayout.Label($"  X: {motorcycleTransform.position.x:F3}");
            GUILayout.Label($"  Y: {motorcycleTransform.position.y:F3}");
            GUILayout.Label($"  Z: {motorcycleTransform.position.z:F3}");
        }

        GUILayout.Label("Trajectories:", new GUIStyle(GUI.skin.label) { fontStyle = FontStyle.Bold });
        GUILayout.Label($"  Real: {realTrajectoryPoints.Count} points");
        GUILayout.Label($"  Predicted: {predictedTrajectoryPoints.Count} points");

        GUILayout.Space(10);
        GUILayout.Label("Controls: Press 'R' to reset trajectories", 
            new GUIStyle(GUI.skin.label) { fontSize = 10 });

        GUILayout.EndArea();
    }

    void Update()
    {
        // Handle keyboard input
        if (Input.GetKeyDown(KeyCode.R))
        {
            ClearTrajectories();
        }

        // Handle connection loss
        if (webSocket != null && (webSocket.ReadyState == WebSocketState.Closing || 
            webSocket.ReadyState == WebSocketState.Closed))
        {
            isConnected = false;
        }
    }

    void ClearTrajectories()
    {
        realTrajectoryPoints.Clear();
        predictedTrajectoryPoints.Clear();
        realTrajectory.positionCount = 0;
        predictedTrajectory.positionCount = 0;
    }

    void OnDestroy()
    {
        if (webSocket != null)
        {
            webSocket.Close();
            webSocket.OnOpen -= null;
            webSocket.OnMessage -= null;
        }
    }

    public void SetMotorcyclePrefab(GameObject prefab)
    {
        motorcyclePrefab = prefab;
        if (motorcycleTransform == null)
        {
            GameObject instance = Instantiate(prefab);
            motorcycleTransform = instance.transform;
        }
    }

    public bool IsConnected => isConnected;

    public int RealTrajectoryPointCount => realTrajectoryPoints.Count;
    public int PredictedTrajectoryPointCount => predictedTrajectoryPoints.Count;
}

/// <summary>
/// Optional: Helper class for scene setup
/// </summary>
public class DigitalTwinManager : MonoBehaviour
{
    [SerializeField] private TelemetryReceiver telemetryReceiver;

    void Start()
    {
        if (telemetryReceiver == null)
            telemetryReceiver = GetComponent<TelemetryReceiver>();

        Debug.Log("Digital Twin Manager initialized");
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.F1))
        {
            Debug.Log($"Connected: {telemetryReceiver.IsConnected} | " +
                     $"Real Trajectory: {telemetryReceiver.RealTrajectoryPointCount} | " +
                     $"Predicted: {telemetryReceiver.PredictedTrajectoryPointCount}");
        }
    }
}
