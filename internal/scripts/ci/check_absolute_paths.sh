#!/bin/bash
# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.

RETURN=0

# ^[[:space:]]* matches start of line followed by optional whitespace
# (?<!["']) ensures we're not inside a string (no quote before the pattern)
# Exclude */cuda/* directories as they contain C++ extension modules that require relative imports
RELATIVE_IMPORT_NRE=$(find nre/ -name '*.py' -not -path '*/cuda/*' -exec grep -n '^[[:space:]]*import \.' {} +)
RELATIVE_FROM_IMPORT_NRE=$(find nre/ -name '*.py' -not -path '*/cuda/*' -exec grep -n '^[[:space:]]*from \.' {} +)

RELATIVE_IMPORT_SCRIPTS=$(find internal/scripts/ -name '*.py' -exec grep -n '^[[:space:]]*import \.' {} +)
RELATIVE_FROM_IMPORT_SCRIPTS=$(find internal/scripts/ -name '*.py' -exec grep -n '^[[:space:]]*from \.' {} +)

if [ -n "$RELATIVE_IMPORT_NRE" ]; then
  echo "Relative import found in nre directory. Please convert to absolute import.\n"
  echo "$RELATIVE_IMPORT_NRE"
  RETURN=1
fi

if [ -n "$RELATIVE_FROM_IMPORT_NRE" ]; then
  echo "Relative from import found in nre directory. Please convert to absolute import.\n"
  echo "$RELATIVE_FROM_IMPORT_NRE"
  RETURN=1
fi

if [ -n "$RELATIVE_IMPORT_SCRIPTS" ]; then
  echo -e "Relative import found in scripts directory. Please convert to absolute import.\n"
  echo "$RELATIVE_IMPORT_SCRIPTS"
  RETURN=1
fi

if [ -n "$RELATIVE_FROM_IMPORT_SCRIPTS" ]; then
  echo -e "Relative from import found in scripts directory. Please convert to absolute import.\n"
  echo "$RELATIVE_FROM_IMPORT_SCRIPTS"
  RETURN=1
fi

if [ $RETURN -eq 0 ]; then
  echo "No relative import found. :)"
fi

exit $RETURN
