#!/bin/sh
# Apply $1 to the CLBlast source tree (cwd). Fails unless the patch applies or
# is already applied. -l: CLBlast sources are CRLF, the patch is LF.
set -e
[ -f "$1" ] || { echo "CLBlast patch missing: $1" >&2; exit 1; }
if patch -p1 -R -l -s -f --dry-run < "$1" >/dev/null 2>&1; then
  echo "CLBlast patch already applied: $1"
else
  patch -p1 -N -l < "$1"
fi
