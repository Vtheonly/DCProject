import os
import sys
import subprocess
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_BUILD_SCRIPT = os.path.join(ROOT_DIR, 'cpp', 'build.py')
APP_PATH = os.path.join(ROOT_DIR, 'src', 'ui', 'app.py')

def check_venv():
    """Verify we are running in a virtual environment."""
    if not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
        print("⚠️  Warning: Not running in a virtual environment.")
        print("   Recommended: source venv/bin/activate")

def build_cpp_core():
    """Build the C++ DSP module if it doesn't exist."""
    so_files = [f for f in os.listdir(ROOT_DIR) if f.startswith('microgrid_dsp') and f.endswith('.so')]
    if not so_files:
        print("🔨 C++ DSP Core not found. Building...")
        try:
            subprocess.run([sys.executable, CPP_BUILD_SCRIPT], check=True, cwd=ROOT_DIR)
            print("✅ Build successful.")
        except subprocess.CalledProcessError:
            print("❌ Build failed. Check C++ dependencies.")
            sys.exit(1)
    else:
        print("✅ C++ DSP Core found.")

def main():
    print("🚀 Starting DC Microgrid Fault Detection System...")
    
    check_venv()
    build_cpp_core()
    
    print("   -> Launching Streamlit UI...")
    try:
        # Launch Streamlit
        env = os.environ.copy()
        env["PYTHONPATH"] = ROOT_DIR
        
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", APP_PATH, "--server.port", "8501", "--server.address", "0.0.0.0"],
            env=env,
            check=True
        )
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user.")
    except Exception as e:
        print(f"\n❌ Error launching system: {e}")

if __name__ == "__main__":
    main()
