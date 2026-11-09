#!/usr/bin/env bash
# Restore the pre-upgrade engine, then use the original launcher and its context arguments.
set -euo pipefail
STRATA_ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
STRATA_SNAPSHOT=/home/devilchen/.local/state/strata-upgrades/20261109-8b4c09bd
"$STRATA_ROOT/.venv/bin/python" - "$STRATA_ROOT" "$STRATA_SNAPSHOT" <<'PY'
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path

root, snapshot = map(Path, sys.argv[1:])
manifest = json.loads((snapshot / 'baseline-sha256.json').read_text())
names = ['engine-cuda12/strata', 'engine-cuda12/libstrata_kq.so', 'engine-cuda12/BUILD.json']
config_name = 'strata-unsloth-ud-iq4_xs.json'
backup_config = snapshot / 'backup' / config_name
if hashlib.sha256(backup_config.read_bytes()).hexdigest() != manifest[config_name]:
    raise SystemExit('Rollback configuration snapshot failed checksum verification')
for name in names:
    source = snapshot / 'backup' / name
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest[name]:
        raise SystemExit('Rollback snapshot failed checksum verification: ' + name)
for name in names:
    source, target = snapshot / 'backup' / name, root / name
    temporary = target.with_name(target.name + '.rollback-' + str(os.getpid()))
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
original = json.loads(backup_config.read_text())
config_path = root / config_name
current = json.loads(config_path.read_text())
# Revert only this upgrade's internal placement/memory tuning. Preserve credentials,
# context changes made through start-strata.sh and every other application setting.
current['layer_split'] = original.get('layer_split', 'auto')
if '--resident-experts' not in original['args']:
    current['args'] = [x for x in current['args'] if x != '--resident-experts']
temporary = config_path.with_name('.strata-rollback-' + str(os.getpid()))
try:
    temporary.write_text(json.dumps(current, indent=1, ensure_ascii=False) + '\n')
    os.chmod(temporary, stat.S_IMODE(config_path.stat().st_mode))
    os.replace(temporary, config_path)
finally:
    temporary.unlink(missing_ok=True)
print('Restored engine 0.1.45.3 and original memory placement; starting with existing API settings.', flush=True)
PY
exec "$STRATA_ROOT/start-strata.sh" "$@"
