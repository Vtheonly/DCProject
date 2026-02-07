import os
import sys
import subprocess

def main():
    print("🚀 Starting DC Microgrid Fault Detection System...")
    print("   -> Launching Streamlit UI...")
    
    # Get the path to app.py
    app_path = os.path.join(os.path.dirname(__file__), 'src', 'ui', 'app.py')
    
    # Launch Streamlit using the current Python executable
    try:
        # Use python -m streamlit run app.py to ensure we use the venv's streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user.")
    except Exception as e:
        print(f"\n❌ Error launching system: {e}")

if __name__ == "__main__":
    main()
