#!/usr/bin/env bash
# Minimal local-only stand-in for Kaldi's utils/run.pl.
#
# Usage: run.pl [options] <log-file> <command> [args...]
#
# Runs <command> locally, tees stdout/stderr to <log-file>, and exits with
# the command's exit status. Unlike the real run.pl this does not support
# JOB=1:N array jobs or queue submission -- there's nothing to submit to here.

while [[ "$1" == --* ]]; do
  shift 2 2>/dev/null || shift 1
done

log_file=$1
shift

mkdir -p "$(dirname "$log_file")"
{
  echo "# $(date)"
  echo "# $*"
  echo "# cwd: $(pwd)"
} > "$log_file"

"$@" 2>&1 | tee -a "$log_file"
exit "${PIPESTATUS[0]}"
