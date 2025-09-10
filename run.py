import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BACKEND_API = ROOT / "SpaceX_ETL" / "Backend" / "api.py"
FRONTEND_APP = ROOT / "SpaceX_ETL" / "Frontend" / "app.py"


def prompt_mode() -> str:
    print("Select mode: \n  1) Offline (Streamlit only)\n  2) Online (start API + Streamlit)")
    choice = input("Enter 1 or 2 [1]: ").strip() or "1"
    return "online" if choice == "2" else "offline"


def main():
    # Optional CLI: python run.py [offline|online]
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else prompt_mode()
    if mode not in {"offline", "online"}:
        print("Invalid mode. Use 'offline' or 'online'.")
        sys.exit(1)

    api_proc = None
    try:
        if mode == "online":
            if not BACKEND_API.exists():
                print("Backend API not found at:", BACKEND_API)
                sys.exit(1)
            print("Starting API at http://127.0.0.1:8000 ...")
            api_proc = subprocess.Popen([sys.executable, str(BACKEND_API)], cwd=str(BACKEND_API.parent))

        if not FRONTEND_APP.exists():
            print("Frontend app not found at:", FRONTEND_APP)
            sys.exit(1)
        print("Launching Streamlit dashboard ...")
        # Streamlit runs in the foreground; when it exits, we clean up the API (if any)
        code = subprocess.call([sys.executable, "-m", "streamlit", "run", str(FRONTEND_APP)], cwd=str(FRONTEND_APP.parent))
        sys.exit(code)
    finally:
        if api_proc is not None:
            try:
                api_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()


