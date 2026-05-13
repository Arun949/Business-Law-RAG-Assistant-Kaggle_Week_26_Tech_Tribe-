"""
launcher.py — Starts the Gradio web UI (and optionally the Signal bot).

Run from project root:
    python app/launcher.py

The script pins the working directory to the project root so every relative
path (data/, prompts/, .env) resolves correctly, regardless of where the
command is invoked from.
"""

import os
import sys
import time
import signal
import socket
import multiprocessing

# ── Resolve paths once, at import time ────────────────────────────────────────
APP_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

GRADIO_PORT  = 7860
GRADIO_URL   = f"http://127.0.0.1:{GRADIO_PORT}"

# Minimum seconds a process must stay alive to be considered "healthy"
# (prevents restart-loop when misconfigured)
_MIN_UPTIME  = 15


# ── Helpers ────────────────────────────────────────────────────────────────────

def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _signal_service_reachable(host: str = "localhost", port: int = 8080,
                               timeout: float = 2.0) -> bool:
    """Return True only if signal-cli-rest-api is actually running (not just any TCP listener)."""
    try:
        import urllib.request
        req = urllib.request.urlopen(
            f"http://{host}:{port}/v1/about", timeout=timeout
        )
        return req.status == 200
    except Exception:
        return False


def _wait_for_port(port: int, timeout: float = 30.0) -> bool:
    """Block until the port is accepting connections (Gradio is up)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


# ── Child process: Gradio UI ──────────────────────────────────────────────────

def run_gradio():
    # Pin CWD and path so every relative import/file-open works
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, APP_DIR)

    from dotenv import load_dotenv
    load_dotenv()

    import gradio as gr
    import gradio_app          # builds demo Blocks (launch guarded by __main__)

    print(f"[Gradio] Starting at {GRADIO_URL} …")
    gradio_app.demo.queue()
    gradio_app.demo.launch(
        server_port=GRADIO_PORT,
        server_name="127.0.0.1",
        share=False,
        theme=gr.themes.Soft(primary_hue="blue"),
        allowed_paths=[os.path.join(PROJECT_ROOT, "data", "sources")],
        quiet=True,            # suppress Gradio's own banner (we print our own)
    )


# ── Child process: Signal bot ─────────────────────────────────────────────────

def run_signal_bot():
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, APP_DIR)

    from dotenv import load_dotenv
    load_dotenv()

    # Guard: refuse to start if phone number is not configured
    phone = os.getenv("SIGNAL_PHONE_NUMBER", "").strip()
    if not phone:
        print("[Signal] SIGNAL_PHONE_NUMBER not set in .env — bot will not start.")
        sys.exit(2)            # exit code 2 = "not configured" (don't restart)

    import signal_bot
    print(f"[Signal] Starting for {phone} …")
    signal_bot.main()


# ── Startup banner ────────────────────────────────────────────────────────────

def _print_banner(signal_enabled: bool):
    sep = "─" * 56
    print(f"\n{sep}")
    print("  🤖  Tech Tribe — Business Law RAG Assistant")
    print(sep)
    print(f"  ✅  Gradio UI  →  {GRADIO_URL}")
    if signal_enabled:
        print("  ✅  Signal bot  →  running")
    else:
        print("  ⏭   Signal bot  →  skipped (signal-cli not reachable)")
    print(f"{sep}")
    print("  Press Ctrl+C to stop all services.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Always run from project root regardless of how the script was invoked
    os.chdir(PROJECT_ROOT)

    # macOS / Python 3.12 default is already "spawn"; force=True is a no-op
    # if it's already set, which prevents the RuntimeError on re-import.
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not _port_free(GRADIO_PORT):
        print(f"[Launcher] ERROR: port {GRADIO_PORT} is already in use.")
        print(f"[Launcher] Kill it first:  lsof -ti:{GRADIO_PORT} | xargs kill")
        sys.exit(1)

    missing = []
    for path in ("data/my_index.faiss", "data/chunks.json", "prompts/v1.txt"):
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print("[Launcher] ERROR: required files missing:")
        for p in missing:
            print(f"           {p}")
        print("[Launcher] Run  python app/build_index.py  first.")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("[Launcher] ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    # ── Launch Gradio ─────────────────────────────────────────────────────────
    gradio_proc       = multiprocessing.Process(target=run_gradio, name="gradio", daemon=True)
    gradio_start_time = time.time()
    gradio_proc.start()

    print("[Launcher] Waiting for Gradio to start…")
    if not _wait_for_port(GRADIO_PORT, timeout=40.0):
        print("[Launcher] ERROR: Gradio did not start within 40 s.")
        gradio_proc.terminate()
        sys.exit(1)

    # ── Launch Signal bot (optional) ──────────────────────────────────────────
    signal_enabled  = _signal_service_reachable()
    signal_proc     = None
    signal_start_t  = None

    if signal_enabled:
        signal_proc    = multiprocessing.Process(target=run_signal_bot, name="signal-bot", daemon=True)
        signal_start_t = time.time()
        signal_proc.start()

    _print_banner(signal_enabled)

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    def _shutdown(signum, frame):
        print("\n[Launcher] Shutting down …")
        gradio_proc.terminate()
        if signal_proc is not None:
            signal_proc.terminate()
            signal_proc.join(timeout=5)
        gradio_proc.join(timeout=5)
        print("[Launcher] Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Watchdog loop ─────────────────────────────────────────────────────────
    while True:
        time.sleep(5)

        # Restart Gradio only if it was healthy for at least _MIN_UPTIME seconds
        if not gradio_proc.is_alive():
            alive_for = time.time() - gradio_start_time
            if alive_for < _MIN_UPTIME:
                print(f"[Launcher] Gradio crashed after {alive_for:.0f}s — not restarting "
                      f"(likely a startup error). Check the output above.")
                sys.exit(1)
            print("[Launcher] Gradio exited unexpectedly — restarting …")
            gradio_proc       = multiprocessing.Process(target=run_gradio, name="gradio", daemon=True)
            gradio_start_time = time.time()
            gradio_proc.start()

        # Restart Signal bot only if it was healthy AND signal-cli is still reachable.
        # exit-code 2 means "not configured" — never restart.
        if signal_proc is not None and not signal_proc.is_alive():
            exit_code  = signal_proc.exitcode
            alive_for  = time.time() - signal_start_t
            if exit_code == 2 or alive_for < _MIN_UPTIME:
                # Not configured or crashed immediately — disable permanently
                print(f"[Launcher] Signal bot exited (code={exit_code}, uptime={alive_for:.0f}s) "
                      f"— disabling restarts.")
                signal_proc = None
            elif _signal_service_reachable():
                print("[Launcher] Signal bot exited unexpectedly — restarting …")
                signal_proc    = multiprocessing.Process(target=run_signal_bot,
                                                         name="signal-bot", daemon=True)
                signal_start_t = time.time()
                signal_proc.start()
            else:
                print("[Launcher] Signal bot exited and signal-cli is unreachable — disabling.")
                signal_proc = None
