#!/bin/bash

################################################################################
#                 MOTO-EDGE-RL COMPLETE TRAINING PIPELINE                     #
#                                                                              #
# This script orchestrates the complete MLOps pipeline for motorcycle racing  #
# RL training and edge deployment.                                            #
#                                                                              #
# Pipeline Steps:                                                             #
#   1. Data Generation: Create synthetic Minari datasets                      #
#   2. Training: Offline pre-training + Online PPO fine-tuning                #
#   3. Deployment: Export model to edge formats (ONNX, TFLite)                #
#                                                                              #
# Usage:                                                                       #
#   ./run_pipeline.sh [OPTIONS]                                               #
#                                                                              #
# Options:                                                                     #
#   --skip-data       Skip data generation (use existing datasets)            #
#   --skip-train      Skip training (use existing model)                      #
#   --skip-export     Skip model export                                       #
#   --laps NUM        Number of laps per rider (default: 100)                 #
#   --timesteps NUM   PPO training timesteps (default: 100000)                #
#   --help            Show this help message                                  #
#                                                                              #
# Requirements:                                                                #
#   - Python 3.8+                                                             #
#   - All dependencies from requirements.txt                                  #
#   - 2GB free disk space (for datasets + models)                             #
#   - ~10 hours total runtime (data: 30min, training: 6h, export: 30min)      #
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python3"
DATA_LAPS=100
PPO_TIMESTEPS=100000
EVAL_EPISODES=10
SEED=42

# Flags
SKIP_DATA=false
SKIP_TRAIN=false
SKIP_EXPORT=false
SKIP_CLEANUP=false

# Directories
DATA_DIR="${SCRIPT_DIR}/data/processed"
MODEL_DIR="${SCRIPT_DIR}/models"
TRAINING_DIR="${SCRIPT_DIR}/src/training"
DEPLOYMENT_DIR="${SCRIPT_DIR}/src/deployment"
DATA_SCRIPT="${SCRIPT_DIR}/src/data/generate_synthetic_data.py"
TRAIN_SCRIPT="${SCRIPT_DIR}/src/training/train_hybrid.py"
EXPORT_SCRIPT="${SCRIPT_DIR}/src/deployment/export_to_edge.py"

# Timestamps for logging
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

################################################################################
# UTILITY FUNCTIONS
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "${LOG_FILE}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_FILE}"
}

print_header() {
    echo -e "\n${BLUE}================================${NC}" | tee -a "${LOG_FILE}"
    echo -e "${BLUE}$1${NC}" | tee -a "${LOG_FILE}"
    echo -e "${BLUE}================================${NC}\n" | tee -a "${LOG_FILE}"
}

print_subheader() {
    echo -e "\n${YELLOW}--- $1 ---${NC}\n" | tee -a "${LOG_FILE}"
}

show_help() {
    head -n 30 "$0" | tail -n 28
}

check_python() {
    if ! command -v "${PYTHON_CMD}" &> /dev/null; then
        log_error "Python3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(${PYTHON_CMD} --version 2>&1 | awk '{print $2}')
    log_info "Python version: ${PYTHON_VERSION}"
}

check_dependencies() {
    log_info "Checking required Python packages..."
    
    local required_packages=("numpy" "gymnasium" "stable_baselines3" "torch" "h5py")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! ${PYTHON_CMD} -c "import ${package}" 2>/dev/null; then
            missing_packages+=("${package}")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        log_warning "Missing packages: ${missing_packages[*]}"
        log_info "Installing dependencies from requirements.txt..."
        pip install -q -r "${SCRIPT_DIR}/requirements.txt" || {
            log_error "Failed to install dependencies"
            exit 1
        }
    else
        log_success "All dependencies are installed"
    fi
}

print_configuration() {
    print_subheader "Pipeline Configuration"
    log_info "Data laps per rider: ${DATA_LAPS}"
    log_info "PPO training timesteps: ${PPO_TIMESTEPS}"
    log_info "Evaluation episodes: ${EVAL_EPISODES}"
    log_info "Random seed: ${SEED}"
    log_info "Skip data generation: ${SKIP_DATA}"
    log_info "Skip training: ${SKIP_TRAIN}"
    log_info "Skip model export: ${SKIP_EXPORT}"
}

################################################################################
# PIPELINE STEPS
################################################################################

step_data_generation() {
    print_header "STEP 1: DATA GENERATION"
    
    if [ "${SKIP_DATA}" = true ]; then
        log_warning "Skipping data generation"
        
        # Check if datasets exist
        if [ ! -f "${DATA_DIR}/pro_rider_dataset.hdf5" ] || [ ! -f "${DATA_DIR}/amateur_rider_dataset.hdf5" ]; then
            log_error "Datasets not found but data generation skipped"
            log_info "Run without --skip-data flag to generate datasets"
            exit 1
        fi
        log_success "Using existing datasets"
        return 0
    fi
    
    if [ ! -f "${DATA_SCRIPT}" ]; then
        log_error "Data generation script not found: ${DATA_SCRIPT}"
        exit 1
    fi
    
    print_subheader "Generating synthetic motorcycle racing data"
    log_info "Creating ${DATA_LAPS} laps of pro and amateur rider data..."
    log_info "Output directory: ${DATA_DIR}"
    
    mkdir -p "${DATA_DIR}"
    
    if ${PYTHON_CMD} "${DATA_SCRIPT}" \
        --laps "${DATA_LAPS}" \
        --output-dir "${DATA_DIR}" \
        --seed "${SEED}" 2>&1 | tee -a "${LOG_FILE}"; then
        log_success "Data generation completed"
        
        # Show dataset sizes
        if [ -f "${DATA_DIR}/pro_rider_dataset.hdf5" ]; then
            PRO_SIZE=$(du -h "${DATA_DIR}/pro_rider_dataset.hdf5" | cut -f1)
            log_info "Pro rider dataset size: ${PRO_SIZE}"
        fi
        if [ -f "${DATA_DIR}/amateur_rider_dataset.hdf5" ]; then
            AMATEUR_SIZE=$(du -h "${DATA_DIR}/amateur_rider_dataset.hdf5" | cut -f1)
            log_info "Amateur rider dataset size: ${AMATEUR_SIZE}"
        fi
    else
        log_error "Data generation failed"
        exit 1
    fi
}

step_training() {
    print_header "STEP 2: HYBRID TRAINING (OFFLINE + ONLINE)"
    
    if [ "${SKIP_TRAIN}" = true ]; then
        log_warning "Skipping training"
        
        if [ ! -f "${MODEL_DIR}/moto_edge_policy.zip" ]; then
            log_error "Model not found but training skipped"
            log_info "Run without --skip-train flag to train the model"
            exit 1
        fi
        log_success "Using existing trained model"
        return 0
    fi
    
    if [ ! -f "${TRAIN_SCRIPT}" ]; then
        log_error "Training script not found: ${TRAIN_SCRIPT}"
        exit 1
    fi
    
    print_subheader "Training hybrid offline-online RL policy"
    log_info "Step 2a: Behavior Cloning pre-training from Minari datasets"
    log_info "Step 2b: PPO fine-tuning for ${PPO_TIMESTEPS} timesteps"
    log_info "Step 2c: Evaluation on ${EVAL_EPISODES} test episodes"
    
    mkdir -p "${MODEL_DIR}"
    
    TRAIN_START_TIME=$(date +%s)
    
    if ${PYTHON_CMD} "${TRAIN_SCRIPT}" \
        --pro-dataset "${DATA_DIR}/pro_rider_dataset.hdf5" \
        --amateur-dataset "${DATA_DIR}/amateur_rider_dataset.hdf5" \
        --output-model "${MODEL_DIR}/moto_edge_policy.zip" \
        --timesteps "${PPO_TIMESTEPS}" \
        --eval-episodes "${EVAL_EPISODES}" \
        --seed "${SEED}" 2>&1 | tee -a "${LOG_FILE}"; then
        
        TRAIN_END_TIME=$(date +%s)
        TRAIN_DURATION=$((TRAIN_END_TIME - TRAIN_START_TIME))
        TRAIN_HOURS=$((TRAIN_DURATION / 3600))
        TRAIN_MINUTES=$(((TRAIN_DURATION % 3600) / 60))
        
        log_success "Training completed in ${TRAIN_HOURS}h ${TRAIN_MINUTES}m"
        
        # Show model info
        if [ -f "${MODEL_DIR}/moto_edge_policy.zip" ]; then
            MODEL_SIZE=$(du -h "${MODEL_DIR}/moto_edge_policy.zip" | cut -f1)
            log_info "Trained model size: ${MODEL_SIZE}"
        fi
    else
        log_error "Training failed"
        exit 1
    fi
}

step_deployment() {
    print_header "STEP 3: MODEL EXPORT FOR EDGE DEPLOYMENT"
    
    if [ "${SKIP_EXPORT}" = true ]; then
        log_warning "Skipping model export"
        return 0
    fi
    
    if [ ! -f "${EXPORT_SCRIPT}" ]; then
        log_error "Export script not found: ${EXPORT_SCRIPT}"
        exit 1
    fi
    
    if [ ! -f "${MODEL_DIR}/moto_edge_policy.zip" ]; then
        log_error "Trained model not found: ${MODEL_DIR}/moto_edge_policy.zip"
        exit 1
    fi
    
    print_subheader "Exporting model for edge devices (ONNX → TensorFlow → TFLite)"
    log_info "Target device: ESP32 microcontroller"
    log_info "Deployment formats:"
    log_info "  - ONNX (cross-platform neural network format)"
    log_info "  - TensorFlow SavedModel (serialized format)"
    log_info "  - TFLite (optimized for edge devices)"
    log_info "  - TFLite Quantized (int8, reduced size)"
    
    mkdir -p "${MODEL_DIR}/edge_deployment"
    
    EXPORT_START_TIME=$(date +%s)
    
    if ${PYTHON_CMD} "${EXPORT_SCRIPT}" \
        --model "${MODEL_DIR}/moto_edge_policy.zip" \
        --output-dir "${MODEL_DIR}/edge_deployment/" 2>&1 | tee -a "${LOG_FILE}"; then
        
        EXPORT_END_TIME=$(date +%s)
        EXPORT_DURATION=$((EXPORT_END_TIME - EXPORT_START_TIME))
        EXPORT_MINUTES=$((EXPORT_DURATION / 60))
        
        log_success "Model export completed in ${EXPORT_MINUTES}m"
        
        # Show exported model sizes
        print_subheader "Exported model file sizes"
        for file in "${MODEL_DIR}/edge_deployment"/*.{onnx,tflite}; do
            if [ -f "$file" ]; then
                SIZE=$(du -h "$file" | cut -f1)
                FILENAME=$(basename "$file")
                log_info "${FILENAME}: ${SIZE}"
            fi
        done
    else
        log_error "Model export failed"
        exit 1
    fi
}

step_summary() {
    print_header "PIPELINE SUMMARY"
    
    print_subheader "Generated Artifacts"
    
    if [ -f "${DATA_DIR}/pro_rider_dataset.hdf5" ]; then
        log_success "✓ Pro rider Minari dataset: ${DATA_DIR}/pro_rider_dataset.hdf5"
    fi
    
    if [ -f "${DATA_DIR}/amateur_rider_dataset.hdf5" ]; then
        log_success "✓ Amateur rider Minari dataset: ${DATA_DIR}/amateur_rider_dataset.hdf5"
    fi
    
    if [ -f "${MODEL_DIR}/moto_edge_policy.zip" ]; then
        log_success "✓ Trained PPO model: ${MODEL_DIR}/moto_edge_policy.zip"
    fi
    
    if [ -f "${MODEL_DIR}/edge_deployment/moto_edge_policy.onnx" ]; then
        log_success "✓ ONNX model: ${MODEL_DIR}/edge_deployment/moto_edge_policy.onnx"
    fi
    
    if [ -f "${MODEL_DIR}/edge_deployment/moto_edge_policy.tflite" ]; then
        log_success "✓ TFLite model: ${MODEL_DIR}/edge_deployment/moto_edge_policy.tflite"
    fi
    
    if [ -f "${MODEL_DIR}/edge_deployment/moto_edge_policy_quantized.tflite" ]; then
        log_success "✓ Quantized TFLite: ${MODEL_DIR}/edge_deployment/moto_edge_policy_quantized.tflite"
    fi
    
    print_subheader "Next Steps"
    log_info "1. Review model performance in: ${MODEL_DIR}/model_metadata.json"
    log_info "2. Deploy TFLite model to ESP32 using TFLite runtime"
    log_info "3. Integrate haptic feedback system (PWM/I2C)"
    log_info "4. Test on-device inference latency and accuracy"
    
    log_info "\nFull pipeline log: ${LOG_FILE}"
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-data)
                SKIP_DATA=true
                shift
                ;;
            --skip-train)
                SKIP_TRAIN=true
                shift
                ;;
            --skip-export)
                SKIP_EXPORT=true
                shift
                ;;
            --laps)
                DATA_LAPS="$2"
                shift 2
                ;;
            --timesteps)
                PPO_TIMESTEPS="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Initialize
    print_header "MOTO-EDGE-RL COMPLETE TRAINING PIPELINE"
    log_info "Pipeline start time: $(date)"
    log_info "Log file: ${LOG_FILE}"
    
    # Pre-flight checks
    check_python
    check_dependencies
    print_configuration
    
    # Execute pipeline
    TOTAL_START_TIME=$(date +%s)
    
    step_data_generation
    step_training
    step_deployment
    step_summary
    
    TOTAL_END_TIME=$(date +%s)
    TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
    TOTAL_HOURS=$((TOTAL_DURATION / 3600))
    TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
    
    # Final status
    echo -e "\n" | tee -a "${LOG_FILE}"
    log_success "COMPLETE PIPELINE FINISHED SUCCESSFULLY!"
    log_info "Total execution time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
    log_info "Pipeline end time: $(date)"
    print_header "🏁 READY FOR EDGE DEPLOYMENT"
    
    return 0
}

# Run main function
main "$@"
exit $?
