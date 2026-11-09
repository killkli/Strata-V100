#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
CONFIG="$ROOT/strata-unsloth-ud-iq4_xs.json"
RUN_SCRIPT="$ROOT/run-unsloth-ud-iq4_xs.sh"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=$(command -v python3)
fi

usage() {
  cat <<'EOF'
Usage: ./start-strata.sh [CONTEXT]

Starts the installed Strata model, keeping the current host and API key.
Without CONTEXT, it keeps the context already saved in the model config.

Examples:
  ./start-strata.sh 131072
  ./start-strata.sh 128k
  ./start-strata.sh 256k

Accepts 8K through 256K in 1K steps. Contexts above the model's native
256K limit need a separate RoPE scaling setup.
EOF
}

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
  fi
fi

CONTEXT=current
if [[ $# -eq 1 ]]; then
  CONTEXT="$1"
fi

"$PYTHON" - "$ROOT" "$CONTEXT" <<'PY'
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

root = Path(sys.argv[1])
requested = sys.argv[2].strip().lower()
config_path = root / "strata-unsloth-ud-iq4_xs.json"
run_script = root / "run-unsloth-ud-iq4_xs.sh"

aliases = {
    "8k": 8192,
    "32k": 32768,
    "64k": 65536,
    "128k": 131072,
    "131k": 131072,
    "200k": 204800,
    "256k": 262144,
}

def parse_context(value):
    if value == "current":
        return None
    if value in aliases:
        result = aliases[value]
    elif value.endswith("k") and value[:-1].isdigit():
        result = int(value[:-1]) * 1024
    else:
        try:
            result = int(value)
        except ValueError as exc:
            raise ValueError("context must be a token count such as 131072 or 128k") from exc
    if result < 8192 or result > 262144 or result % 1024:
        raise ValueError("context must be 8K–256K and a multiple of 1K")
    return result

def listeners(port):
    if shutil.which("ss") is None:
        raise RuntimeError("the ss command is required to safely restart the server")
    result = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("could not inspect listening ports with ss")
    found = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[0] != "LISTEN":
            continue
        if fields[3].rsplit(":", 1)[-1] != str(port):
            continue
        pids = set(re.findall(r"pid=(\d+)", line))
        if not pids:
            raise RuntimeError(f"port {port} is occupied, but its process could not be identified")
        for pid in pids:
            found[int(pid)] = line
    return found

def is_this_strata(pid):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return str(root / "serve" / "server.py") in command and str(config_path) in command

def process_alive(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return fields[2] != "Z"
    except (OSError, IndexError):
        return False

def atomic_write(data, mode):
    fd, name = tempfile.mkstemp(prefix=".strata-config-", dir=config_path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(name, mode)
        os.replace(name, config_path)
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise

def main():
    target = parse_context(requested)
    if not config_path.is_file() or not run_script.is_file():
        raise RuntimeError("the installed UD-IQ4_XS config or run script is missing")

    original = config_path.read_bytes()
    mode = stat.S_IMODE(config_path.stat().st_mode)
    config = json.loads(original)
    args = config.get("args")
    if not isinstance(args, list) or "--max-context" not in args[:-1]:
        raise RuntimeError("the saved config has no --max-context setting")
    old_context = int(args[args.index("--max-context") + 1])
    if target is None:
        target = old_context
    if not config.get("api_key"):
        raise RuntimeError("the saved config has no API key; refusing to start without the current key")

    port = int(config.get("port", 8080))
    active = listeners(port)
    unrelated = [pid for pid in active if not is_this_strata(pid)]
    if unrelated:
        raise RuntimeError(f"port {port} is occupied by another process (PID {unrelated[0]}); no changes made")

    changed = target != old_context
    if changed:
        args[args.index("--max-context") + 1] = str(target)
        encoded = (json.dumps(config, indent=1, ensure_ascii=False) + "\n").encode()
        atomic_write(encoded, mode)

    print(f"[start-strata] context: {old_context:,} -> {target:,} tokens", flush=True)
    print(f"[start-strata] host {config.get('host', '127.0.0.1')}:{port}; saved API key preserved", flush=True)

    if active:
        pids = list(active)
        print("[start-strata] stopping the current Strata server gracefully...", flush=True)
        for pid in pids:
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if not listeners(port) and not any(process_alive(pid) for pid in pids):
                break
            time.sleep(0.5)
        else:
            atomic_write(original, mode)
            raise RuntimeError("the current server did not stop within 60 seconds; the old config was restored")

if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"start-strata: {exc}", file=sys.stderr)
        sys.exit(2)
PY

exec "$RUN_SCRIPT"
