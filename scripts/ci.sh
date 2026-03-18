#!/bin/sh
# Archipelag.io Island — Local CI Pipeline
set -e

BOLD='\033[1m'
GREEN='\033[32m'
RED='\033[31m'
RESET='\033[0m'
YELLOW='\033[33m'
PASS="${GREEN}PASS${RESET}"
FAIL="${RED}FAIL${RESET}"
WARN="${YELLOW}WARN${RESET}"

results=""
failed=0

run_step() {
  name="$1"
  shift
  printf "${BOLD}[CI]${RESET} %-30s" "$name"
  if output=$("$@" 2>&1); then
    printf " $PASS\n"
    results="$results\n  $PASS  $name"
  else
    printf " $FAIL\n"
    echo "$output" | tail -20
    results="$results\n  $FAIL  $name"
    failed=1
  fi
}

# Like run_step but doesn't set failed=1 on failure
run_step_warn() {
  name="$1"
  shift
  printf "${BOLD}[CI]${RESET} %-30s" "$name"
  if output=$("$@" 2>&1); then
    printf " $PASS\n"
    results="$results\n  $PASS  $name"
  else
    printf " $WARN\n"
    echo "$output" | tail -10
    results="$results\n  $WARN  $name"
  fi
}

printf "\n${BOLD}Archipelag.io Island — Local CI${RESET}\n\n"

run_step "Format check"       cargo fmt -- --check
run_step "Clippy"             cargo clippy -- -D warnings
run_step "Tests (default)"    cargo test
run_step "Build (release)"    cargo build --release
run_step "Doc check"          cargo doc --no-deps

if command -v cargo-audit >/dev/null 2>&1; then
  run_step_warn "Dependency audit"   cargo audit
else
  printf "${BOLD}[CI]${RESET} %-30s SKIP (cargo-audit not installed)\n" "Dependency audit"
  results="$results\n  SKIP  Dependency audit"
fi

printf "\n${BOLD}Results:${RESET}"
printf "$results\n\n"

if [ "$failed" -eq 1 ]; then
  printf "${RED}${BOLD}CI FAILED${RESET}\n"
  exit 1
else
  printf "${GREEN}${BOLD}CI PASSED${RESET}\n"
fi
