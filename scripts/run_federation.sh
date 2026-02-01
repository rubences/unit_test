#!/bin/bash
# Federated Learning Orchestration Script
#
# Launches a Flower server and 3 clients in parallel to simulate
# federated training across pilots with different driving styles.
#
# Usage:
#   bash scripts/run_federation.sh [rounds] [host] [port]
#
# Example:
#   bash scripts/run_federation.sh 5 localhost 8080

set -e

# Configuration
ROUNDS=${1:-5}
HOST=${2:-localhost}
PORT=${3:-8080}
MIN_CLIENTS=3
OUTPUT_DIR="outputs"
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Federated Learning Experiment"
echo "=========================================="
echo "Rounds: $ROUNDS"
echo "Server: $HOST:$PORT"
echo "Clients: $MIN_CLIENTS (pilots)"
echo "Output: $OUTPUT_DIR"
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "Shutting down clients..."
    pkill -P $$ || true
    sleep 2
    echo "Cleanup complete"
}

trap cleanup EXIT

# Start server in background
echo "[1/4] Starting Federated Learning Server..."
python src/federated/federated_server.py \
    --host "$HOST" \
    --port "$PORT" \
    --rounds "$ROUNDS" \
    --min-available-clients "$MIN_CLIENTS" \
    --output "$OUTPUT_DIR/federated_metrics.json" \
    > "$OUTPUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wait for server to start
sleep 3

# Start clients in parallel
echo "[2/4] Starting Federated Learning Clients..."
for i in {1..3}; do
    echo "  Launching Client $i (Pilot with style $i)..."
    python src/federated/federated_client.py \
        --client-id "$i" \
        --server-address "$HOST:$PORT" \
        > "$OUTPUT_DIR/client_$i.log" 2>&1 &
    CLIENT_PID=$!
    echo "    Client $i PID: $CLIENT_PID"
    sleep 1
done

# Wait for server to finish
echo "[3/4] Waiting for server to complete training..."
wait $SERVER_PID
SERVER_EXIT=$?

# Allow clients to finish gracefully
sleep 2
pkill -P $$ || true
sleep 1

echo "[4/4] Generating convergence comparison plot..."
python scripts/plot_federated_convergence.py \
    --output "$OUTPUT_DIR/convergence_comparison.pdf"

echo ""
echo "=========================================="
echo "Federated Learning Complete!"
echo "=========================================="
echo "Outputs:"
echo "  - Server metrics:     $OUTPUT_DIR/federated_metrics.json"
echo "  - Client 1 losses:    $OUTPUT_DIR/client_1_loss.json"
echo "  - Client 2 losses:    $OUTPUT_DIR/client_2_loss.json"
echo "  - Client 3 losses:    $OUTPUT_DIR/client_3_loss.json"
echo "  - Comparison plot:    $OUTPUT_DIR/convergence_comparison.pdf"
echo "  - Server log:         $OUTPUT_DIR/server.log"
echo "  - Client logs:        $OUTPUT_DIR/client_*.log"
echo ""

exit $SERVER_EXIT
