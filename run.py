import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import webbrowser
import threading
import time


ROOT = Path(__file__).resolve().parent
BACKEND_API = ROOT / "SpaceX_ETL" / "Backend" / "api.py"
FRONTEND_APP = ROOT / "SpaceX_ETL" / "Frontend" / "app.py"


def open_browser_delayed():
    """Open browser after Streamlit starts (5 second delay)"""
    time.sleep(5)
    try:
        webbrowser.open("http://localhost:8501")
    except:
        pass


def prompt_mode_gui() -> str:
    choice = {"value": None}

    def set_and_close(val: str):
        choice["value"] = val
        root.destroy()

    root = tk.Tk()
    root.title("SpaceX ETL Launcher")
    root.geometry("360x160")
    root.resizable(False, False)

    frm = ttk.Frame(root, padding=16)
    frm.pack(fill="both", expand=True)

    title = ttk.Label(frm, text="Choose how to run:", font=("Segoe UI", 11, "bold"))
    title.pack(pady=(0, 12))

    btns = ttk.Frame(frm)
    btns.pack()

    ttk.Button(btns, text="Offline (Streamlit)", command=lambda: set_and_close("offline"), width=22).grid(row=0, column=0, padx=6, pady=6)
    ttk.Button(btns, text="Online (API + Streamlit)", command=lambda: set_and_close("online"), width=22).grid(row=0, column=1, padx=6, pady=6)
    ttk.Button(frm, text="API only", command=lambda: set_and_close("api"), width=22).pack(pady=(8, 0))

    root.mainloop()
    return choice["value"] or "offline"


def main():
    # Optional CLI: python run.py [offline|online|api]
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else prompt_mode_gui()
    if mode not in {"offline", "online", "api"}:
        print("Invalid mode. Use 'offline', 'online', or 'api'.")
        sys.exit(1)

    api_proc = None
    try:
        if mode in ("online", "api"):
            if not BACKEND_API.exists():
                print("Backend API not found at:", BACKEND_API)
                sys.exit(1)
            print("Starting API at http://127.0.0.1:8000 ...")
            api_proc = subprocess.Popen([sys.executable, str(BACKEND_API)], cwd=str(BACKEND_API.parent))

        if mode in ("online", "offline"):
            if not FRONTEND_APP.exists():
                print("Frontend app not found at:", FRONTEND_APP)
                sys.exit(1)
            print("Launching Streamlit dashboard ...")
            
            # Start browser thread (only for GUI mode, not CLI)
            if len(sys.argv) == 1:  # GUI mode
                browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
                browser_thread.start()
            
            # Streamlit with headless mode to prevent auto-browser opening
            code = subprocess.call([
                sys.executable, "-m", "streamlit", "run", str(FRONTEND_APP),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false",
                "--server.port", "8501"
            ], cwd=str(FRONTEND_APP.parent))
            sys.exit(code)
        else:
            # API-only mode: wait for CTRL+C
            print("API started. Press Ctrl+C to stop.")
            api_proc.wait()
    finally:
        if api_proc is not None:
            try:
                api_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()