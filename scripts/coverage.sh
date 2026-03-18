#!/bin/sh
# Generate test coverage report
set -e

if ! command -v cargo-llvm-cov >/dev/null 2>&1; then
  echo '> Installing cargo-llvm-cov...'
  cargo install cargo-llvm-cov
  rustup component add llvm-tools-preview
fi

if [ "$1" = "--html" ]; then
  cargo llvm-cov --html
  open target/llvm-cov/html/index.html 2>/dev/null \
    || xdg-open target/llvm-cov/html/index.html 2>/dev/null \
    || echo "Report at target/llvm-cov/html/index.html"
else
  cargo llvm-cov --text
fi
