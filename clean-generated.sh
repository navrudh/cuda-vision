#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WORKDIR=$(cat "$DIR/project/resources/config/user-ganesh.json" | python -c "import json,sys;obj=json.load(sys.stdin);print(obj['workdir']);")

rm -rf "$WORKDIR/generated"
