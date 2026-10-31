#!/bin/sh
# usage: push_dlls.sh [NAME.dll ...]
# Copies exactly the named DLLs from C:\Windows\System32 into the VPS rootfs and lists each one in
# needed_dlls.txt (once). Then runs the loader probe on one plugin and prints the DLLs the loader
# still reports missing (a DLL is only ever copied after a probe named it).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
for n in "$@"; do
  grep -qix "$n" "$HERE/needed_dlls.txt" 2>/dev/null || echo "$n" >> "$HERE/needed_dlls.txt"
done
{
  if [ $# -gt 0 ]; then (cd /c/Windows/System32 && tar --force-local -cf - "$@"); else tar --force-local -cf - -T /dev/null; fi
} | ssh -o BatchMode=yes compute '
H=~/twixtor-parity/harness; RF=$H/rootfs/x8664_windows; tar xf - -C $RF/Windows/System32 2>&1 | grep -v "in the future" || true
PY=~/twixtor-parity/venv/bin/python; cd $H
timeout 600 $PY explore_load.py $RF $RF/plugins/Twixtor_64.aex > probe.log 2>&1 || true
sed "s/\x1b\[[0-9;]*m//g" probe.log | grep -E "Could not find|Traceback|LOADED|FAILED" | sed "s#$HOME/twixtor-parity/harness/rootfs/x8664_windows/Windows/System32/##" | sort | uniq -c | head -40
sed "s/\x1b\[[0-9;]*m//g" probe.log | grep -c "not implemented" || true'
