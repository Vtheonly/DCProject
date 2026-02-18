#!/bin/bash
# run.sh — Launch the DC Microgrid Fault Detection Platform

# 1. Activate Virtual Environment
source venv/bin/activate

# 2. Build C++ DSP Module (if not already built)
if [ ! -f "microgrid_dsp.cpython-312-x86_64-linux-gnu.so" ]; then
    echo "🔨 Building C++ DSP Core..."
    python3 cpp/build.py
fi

# 3. Launch Streamlit Application
echo "🚀 Launching DC Microgrid Platform..."
PYTHONPATH=. streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0
