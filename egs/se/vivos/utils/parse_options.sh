#!/usr/bin/env bash
# Simplified stand-in for Kaldi's utils/parse_options.sh.
#
# Parses "--some-option value" style args into shell variables of the same
# name (dashes -> underscores), but only for variables that already have a
# default assigned earlier in the calling script -- this mirrors Kaldi's
# convention where a script's options are exactly its pre-declared defaults.
#
# Usage (from a script, after declaring defaults):
#   . utils/parse_options.sh

ret_args=()

while [ $# -gt 0 ]; do
  case "$1" in
    --*)
      name=$(echo "${1#--}" | sed 's/-/_/g')
      if [ "$(eval echo \${${name}+set})" != "set" ]; then
        echo "$0: unknown option: $1" >&2
        exit 1
      fi
      eval "${name}=\"$2\""
      shift 2
      ;;
    *)
      ret_args+=("$1")
      shift
      ;;
  esac
done

set -- "${ret_args[@]}"
