#!/usr/bin/env bash
set -euo pipefail
if detect-secrets scan --all-files | grep -q '"has_secrets": true'; then
  echo 'Secrets found'
  exit 4
fi
echo 'OK'
