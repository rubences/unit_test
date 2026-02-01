#!/usr/bin/env bash

# 🏍️ Bio-Adaptive Haptic Coaching System
# Quick Launch Script

set -e  # Exit on error

echo "🏍️  BIO-ADAPTIVE HAPTIC COACHING SYSTEM"
echo "=================================="
echo ""

# Detect project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

if [ ! -d "$PROJECT_ROOT/moto_bio_project" ]; then
    echo "❌ Error: moto_bio_project not found in $PROJECT_ROOT"
    exit 1
fi

echo "📁 Project root: $PROJECT_ROOT"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✅ $PYTHON_VERSION"
echo ""

# Menu
echo "Select execution mode:"
echo "1) Automated Deployment (run_deployment.py)"
echo "2) Interactive Notebook (Jupyter)"
echo "3) Manual Python Execution"
echo "4) View Reports"
echo "5) Clean Artifacts"
echo ""
read -p "Choose option (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting automated deployment..."
        echo ""
        cd "$PROJECT_ROOT"
        python3 run_deployment.py
        ;;
    
    2)
        echo ""
        echo "📚 Starting Jupyter notebook..."
        echo ""
        cd "$PROJECT_ROOT/moto_bio_project"
        jupyter notebook notebooks/analysis.ipynb
        ;;
    
    3)
        echo ""
        echo "🔧 Manual execution mode"
        echo ""
        cd "$PROJECT_ROOT/moto_bio_project"
        
        echo "⏳ Running data generation..."
        python3 src/data_gen.py
        
        echo "⏳ Training model..."
        python3 src/train.py
        
        echo "⏳ Generating visualizations..."
        python3 src/visualize.py
        
        echo "✅ Manual execution complete"
        ;;
    
    4)
        echo ""
        echo "📊 Reports:"
        echo ""
        
        REPORTS_DIR="$PROJECT_ROOT/moto_bio_project/reports"
        if [ -d "$REPORTS_DIR" ]; then
            if [ -f "$REPORTS_DIR/DEPLOYMENT_SUMMARY.txt" ]; then
                echo "=== Latest Summary ==="
                cat "$REPORTS_DIR/DEPLOYMENT_SUMMARY.txt"
                echo ""
            fi
            
            LATEST_JSON=$(ls -t "$REPORTS_DIR"/*.json 2>/dev/null | head -1)
            if [ -n "$LATEST_JSON" ]; then
                echo "=== Latest JSON Report ==="
                python3 -m json.tool "$LATEST_JSON" | head -50
                echo "..."
            fi
        else
            echo "⚠️  No reports found. Run deployment first."
        fi
        ;;
    
    5)
        echo ""
        read -p "⚠️  Delete all artifacts? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            echo "🗑️  Cleaning..."
            rm -rf "$PROJECT_ROOT/moto_bio_project/logs"/*
            rm -rf "$PROJECT_ROOT/moto_bio_project/models"/*
            rm -rf "$PROJECT_ROOT/moto_bio_project/data"/*
            rm -rf "$PROJECT_ROOT/moto_bio_project/reports"/*
            echo "✅ Cleaned"
        fi
        ;;
    
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "✅ Done"
