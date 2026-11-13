#!/usr/bin/env bash
# Restore a verified engine snapshot while keeping credentials and context changes.
set -euo pipefail
STRATA_ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
STRATA_SNAPSHOT=${STRATA_SNAPSHOT:-/home/devilchen/.local/state/strata-upgrades/20261113-upstream-perf}
STRATA_ROLLBACK_MODE=restore
STRATA_ROLLBACK_START=1
case "${1:-}" in
  --check) STRATA_ROLLBACK_MODE=check; STRATA_ROLLBACK_START=0; shift ;;
  --no-start) STRATA_ROLLBACK_START=0; shift ;;
  -h|--help)
    printf '%s\n' 'Usage: ./rollback-strata-upgrade.sh [--check|--no-start] [CONTEXT]' \
      'STRATA_SNAPSHOT selects an older snapshot. --check changes nothing.' \
      'The default restores the engine and restarts with the saved API settings.'
    exit 0 ;;
esac
"$STRATA_ROOT/.venv/bin/python" - "$STRATA_ROOT" "$STRATA_SNAPSHOT" "$STRATA_ROLLBACK_MODE" <<'PY'
import hashlib
import json
import os
import shutil
import signal
import stat
import sys
import time
from pathlib import Path

root, snapshot = map(Path, sys.argv[1:3])
check_only = sys.argv[3] == 'check'
manifest = json.loads((snapshot / 'baseline-sha256.json').read_text())
known = ['engine-cuda12/strata', 'engine-cuda12/libstrata_kq.so',
         'engine-cuda12/strata-vision', 'engine-cuda12/BUILD.json']
required = set(known) - {'engine-cuda12/strata-vision'}
if not required.issubset(manifest):
    raise SystemExit('Rollback snapshot is missing required engine files')
names = [name for name in known if name in manifest]
config_name = 'strata-unsloth-ud-iq4_xs.json'
backup_config = snapshot / 'backup' / config_name
for name in [*names, config_name]:
    source = snapshot / 'backup' / name
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest[name]:
        raise SystemExit('Rollback snapshot failed checksum verification: ' + name)
version = json.loads((snapshot / 'backup/engine-cuda12/BUILD.json').read_text()).get('version', 'unknown')
if check_only:
    print(f'Rollback snapshot verified: engine {version}, {len(names)} engine files; no changes made.')
    raise SystemExit(0)

config_path = root / config_name
engine_path = (root / 'engine-cuda12/strata').resolve()

def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except (OSError, IndexError):
        return False

def active_engines():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            executable = str((proc / 'exe').resolve()).removesuffix(' (deleted)')
            if executable == str(engine_path) and alive(int(proc.name)):
                found.append(int(proc.name))
        except OSError:
            pass
    return found

managed = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        command = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        server = str(root / 'serve/server.py') in command or (
            'serve.server' in command and (proc / 'cwd').resolve() == root)
        if not server:
            continue
        config = None
        if '--config' in command:
            config = command[command.index('--config') + 1]
        else:
            config = next((x.split('=', 1)[1] for x in command if x.startswith('--config=')), None)
        if config and ((proc / 'cwd').resolve() / config).resolve() == config_path.resolve():
            managed.append(int(proc.name))
    except (OSError, IndexError):
        continue
if active_engines() and not managed:
    raise SystemExit('The installed engine is in use outside its saved server; stop that engine before rollback.')
for pid in managed:
    os.kill(pid, signal.SIGTERM)
if managed:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if not any(alive(pid) for pid in managed) and not active_engines():
            break
        time.sleep(0.25)
    else:
        raise SystemExit('The saved server did not stop; engine files and config were left unchanged.')

# Stage every file before replacing any installed component.
pending = []
try:
    for name in names:
        source, target = snapshot / 'backup' / name, root / name
        temporary = target.with_name(target.name + '.rollback-' + str(os.getpid()))
        shutil.copy2(source, temporary)
        pending.append((temporary, target))
    for temporary, target in pending:
        os.replace(temporary, target)
finally:
    for temporary, _ in pending:
        temporary.unlink(missing_ok=True)

original = json.loads(backup_config.read_text())
current = json.loads(config_path.read_text())
# These are placement settings changed by the local upgrades. Keep the key,
# network settings, context, model choice, and unrelated later edits.
current['layer_split'] = original.get('layer_split', 'auto')
if '--resident-experts' not in original['args']:
    current['args'] = [x for x in current['args'] if x != '--resident-experts']
env = dict(current.get('env', {}))
for name in ('STRATA_HOST_NUMA_NODE', 'STRATA_NUMA_STAGE_LOCAL'):
    if name in original.get('env', {}):
        env[name] = original['env'][name]
    else:
        env.pop(name, None)
current['env'] = env
temporary = config_path.with_name('.strata-rollback-' + str(os.getpid()))
try:
    temporary.write_text(json.dumps(current, indent=1, ensure_ascii=False) + '\n')
    os.chmod(temporary, stat.S_IMODE(config_path.stat().st_mode))
    os.replace(temporary, config_path)
finally:
    temporary.unlink(missing_ok=True)
print(f'Restored engine {version} and saved placement; credentials and context preserved.', flush=True)
PY
if [[ "$STRATA_ROLLBACK_START" == 1 ]]; then
  exec "$STRATA_ROOT/start-strata.sh" "$@"
fi
