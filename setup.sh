#!/bin/bash

# Hospital Length of Stay Prediction - Setup Script
# This script sets up the project environment and runs initial tests

echo "🏥 Hospital Length of Stay Prediction - Setup"
echo "============================================="
echo "DISCLAIMER: This is a research demonstration project and is NOT intended for clinical use."
echo "============================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.10+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "🛠️  Installing development dependencies..."
pip install pytest pytest-cov black ruff mypy pre-commit

# Setup pre-commit hooks
echo "🔗 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p models results assets logs

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/ -v

# Run the main script
echo "🚀 Running main demonstration script..."
python 0452.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Run the interactive demo: streamlit run demo/app.py"
echo "2. Train with custom config: python scripts/train.py --config configs/default.yaml"
echo "3. Evaluate a model: python scripts/evaluate.py --model-path models/best_model.pkl"
echo "4. View documentation: README.md"
echo ""
echo "⚠️  Remember: This is a research demonstration project."
echo "   For clinical use, external validation and regulatory approval are required."
