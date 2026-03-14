#!/bin/sh
# Archipelag.io Node Agent Installer
# Usage: curl -fsSL https://archipelag.io/install.sh | sh
#
# Environment variables:
#   ARCHIPELAG_VERSION  - Version to install (default: latest)
#   ARCHIPELAG_DIR      - Install directory (default: ~/.archipelag)
#   SKIP_DOCKER         - Set to 1 to skip Docker setup

set -e

REPO="archipelag-io/node-agent"
INSTALL_DIR="${ARCHIPELAG_DIR:-$HOME/.archipelag}"
BIN_DIR="$INSTALL_DIR/bin"
CONFIG_DIR="$INSTALL_DIR"
BINARY_NAME="archipelag-agent"

# Colors (disabled if not a terminal)
if [ -t 1 ]; then
    BOLD="\033[1m"
    GREEN="\033[32m"
    YELLOW="\033[33m"
    RED="\033[31m"
    RESET="\033[0m"
else
    BOLD="" GREEN="" YELLOW="" RED="" RESET=""
fi

info()  { printf "${GREEN}>${RESET} %s\n" "$1"; }
warn()  { printf "${YELLOW}!${RESET} %s\n" "$1"; }
error() { printf "${RED}x${RESET} %s\n" "$1" >&2; exit 1; }

# Detect OS and architecture
detect_platform() {
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)

    case "$OS" in
        linux)  OS_NAME="linux" ;;
        darwin) OS_NAME="darwin" ;;
        *)      error "Unsupported operating system: $OS" ;;
    esac

    case "$ARCH" in
        x86_64|amd64)   ARCH_NAME="x86_64" ;;
        aarch64|arm64)  ARCH_NAME="aarch64" ;;
        *)              error "Unsupported architecture: $ARCH" ;;
    esac

    PLATFORM="${OS_NAME}-${ARCH_NAME}"
}

# Get latest version from GitHub releases
get_latest_version() {
    if [ -n "$ARCHIPELAG_VERSION" ]; then
        VERSION="$ARCHIPELAG_VERSION"
        return
    fi

    info "Checking latest version..."
    VERSION=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
        | grep '"tag_name"' | head -1 | sed 's/.*"v\(.*\)".*/\1/')

    if [ -z "$VERSION" ]; then
        error "Could not determine latest version. Set ARCHIPELAG_VERSION manually."
    fi
}

# Download and install the binary
download_binary() {
    ASSET_NAME="${BINARY_NAME}-${VERSION}-${PLATFORM}"
    URL="https://github.com/${REPO}/releases/download/v${VERSION}/${ASSET_NAME}"

    info "Downloading ${BINARY_NAME} v${VERSION} for ${PLATFORM}..."

    mkdir -p "$BIN_DIR"

    HTTP_CODE=$(curl -fsSL -w '%{http_code}' -o "$BIN_DIR/$BINARY_NAME" "$URL")
    if [ "$HTTP_CODE" != "200" ]; then
        rm -f "$BIN_DIR/$BINARY_NAME"
        error "Download failed (HTTP $HTTP_CODE). Check that v${VERSION} has a release for ${PLATFORM}."
    fi

    chmod +x "$BIN_DIR/$BINARY_NAME"
    info "Installed to $BIN_DIR/$BINARY_NAME"
}

# Verify the binary runs
verify_binary() {
    if "$BIN_DIR/$BINARY_NAME" --help >/dev/null 2>&1; then
        info "Binary verified OK"
    else
        # macOS Gatekeeper may block it
        if [ "$OS_NAME" = "darwin" ]; then
            warn "macOS blocked the binary. Removing quarantine flag..."
            xattr -d com.apple.quarantine "$BIN_DIR/$BINARY_NAME" 2>/dev/null || true
            if "$BIN_DIR/$BINARY_NAME" --help >/dev/null 2>&1; then
                info "Binary verified OK (quarantine removed)"
            else
                warn "Binary may still be blocked. Open System Settings > Privacy & Security to allow it."
            fi
        else
            error "Binary failed to run. Try downloading manually from: https://github.com/${REPO}/releases"
        fi
    fi
}

# Check for and install Docker if needed
setup_docker() {
    if [ "$SKIP_DOCKER" = "1" ]; then
        warn "Skipping Docker setup (SKIP_DOCKER=1)"
        DOCKER_AVAILABLE=false
        return
    fi

    # Check if Docker is already available
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        info "Docker is already running"
        DOCKER_AVAILABLE=true
        return
    fi

    if command -v docker >/dev/null 2>&1; then
        warn "Docker is installed but not running"
    else
        info "Docker not found"
    fi

    if [ "$OS_NAME" = "darwin" ]; then
        setup_docker_macos
    elif [ "$OS_NAME" = "linux" ]; then
        setup_docker_linux
    fi
}

setup_docker_macos() {
    # Prefer Colima (lightweight, free) over Docker Desktop (heavy, license required for commercial use)
    if command -v colima >/dev/null 2>&1; then
        info "Starting Colima..."
        colima start --cpu 2 --memory 4 2>/dev/null && DOCKER_AVAILABLE=true && return
    fi

    # Check if Homebrew is available
    if ! command -v brew >/dev/null 2>&1; then
        warn "Docker is required for container workloads."
        warn "Install Homebrew first: https://brew.sh"
        warn "Then run: brew install colima docker && colima start"
        warn "The agent will run in WASM-only mode for now."
        DOCKER_AVAILABLE=false
        return
    fi

    info "Installing Docker runtime via Colima (lightweight, no Docker Desktop needed)..."

    # Install docker CLI and colima
    if ! command -v docker >/dev/null 2>&1; then
        info "Installing Docker CLI..."
        brew install docker 2>&1 | tail -1
    fi

    if ! command -v colima >/dev/null 2>&1; then
        info "Installing Colima..."
        brew install colima 2>&1 | tail -1
    fi

    info "Starting Colima VM (this may take a minute on first run)..."
    if colima start --cpu 2 --memory 4 2>&1 | tail -3; then
        info "Docker is ready (via Colima)"
        DOCKER_AVAILABLE=true
    else
        warn "Colima failed to start. The agent will run in WASM-only mode."
        warn "Try manually: colima start"
        DOCKER_AVAILABLE=false
    fi
}

setup_docker_linux() {
    # On Linux, we bundle crun (lightweight OCI runtime) — no Docker needed
    info "Setting up bundled container runtime (crun)..."

    # Map architecture for crun binary name
    case "$ARCH_NAME" in
        x86_64)   CRUN_PLATFORM="linux-x86_64" ;;
        aarch64)  CRUN_PLATFORM="linux-aarch64" ;;
        *)        warn "No bundled crun for $ARCH_NAME"; DOCKER_AVAILABLE=false; return ;;
    esac

    CRUN_URL="https://github.com/${REPO}/releases/download/v${VERSION}/crun-${CRUN_PLATFORM}"
    CRUN_PATH="$BIN_DIR/crun"

    if [ -f "$CRUN_PATH" ]; then
        info "crun already installed"
        DOCKER_AVAILABLE=true
        return
    fi

    info "Downloading crun for ${CRUN_PLATFORM}..."
    if curl -fsSL -o "$CRUN_PATH" "$CRUN_URL"; then
        chmod +x "$CRUN_PATH"
        info "crun installed to $CRUN_PATH"
        DOCKER_AVAILABLE=true
    else
        warn "Failed to download crun. Container workloads won't be available."
        warn "The agent will still run WASM workloads."
        DOCKER_AVAILABLE=false
    fi
}

# Create default config if none exists
create_config() {
    CONFIG_FILE="$CONFIG_DIR/config.toml"

    if [ -f "$CONFIG_FILE" ]; then
        info "Config already exists at $CONFIG_FILE"
        return
    fi

    HOSTNAME=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo "my-host")

    info "Creating default config at $CONFIG_FILE"
    cat > "$CONFIG_FILE" << EOF
# Archipelag.io Node Agent Configuration
# Docs: https://github.com/${REPO}

[host]
name = "${HOSTNAME}"
region = "auto"

[coordinator]
nats_url = "nats://nats.archipelag.io:4222"

[workload]
llm_chat_image = "ghcr.io/archipelag-io/llm-chat:latest"
gpu_devices = []

[workload.resource_limits]
memory_mb = 4096
read_only_rootfs = true
tmpfs_size_mb = 256
network_disabled = true

[cache]
enable_preload = false
max_cached_images = 10

[signing]
enabled = true
require_signature = false

[registry]
enabled = true
require_digest = false
EOF
}

# Add to PATH
setup_path() {
    SHELL_NAME=$(basename "$SHELL" 2>/dev/null || echo "sh")
    EXPORT_LINE="export PATH=\"$BIN_DIR:\$PATH\""

    # Check if already in PATH
    case ":$PATH:" in
        *":$BIN_DIR:"*) return ;;
    esac

    case "$SHELL_NAME" in
        zsh)  RC_FILE="$HOME/.zshrc" ;;
        bash) RC_FILE="$HOME/.bashrc" ;;
        fish)
            EXPORT_LINE="fish_add_path $BIN_DIR"
            RC_FILE="$HOME/.config/fish/config.fish"
            ;;
        *)    RC_FILE="$HOME/.profile" ;;
    esac

    if [ -f "$RC_FILE" ] && grep -q "$BIN_DIR" "$RC_FILE" 2>/dev/null; then
        return
    fi

    # Create parent directory if needed (e.g. ~/.config/fish/)
    mkdir -p "$(dirname "$RC_FILE")"

    info "Adding $BIN_DIR to PATH in $RC_FILE"
    echo "" >> "$RC_FILE"
    echo "# Archipelag.io" >> "$RC_FILE"
    echo "$EXPORT_LINE" >> "$RC_FILE"
}

# Print next steps
print_success() {
    printf "\n"
    printf "${BOLD}${GREEN}Archipelag.io Node Agent v${VERSION} installed!${RESET}\n"
    printf "\n"
    printf "  Binary:  %s\n" "$BIN_DIR/$BINARY_NAME"
    printf "  Config:  %s\n" "$CONFIG_DIR/config.toml"

    if [ "$DOCKER_AVAILABLE" = "true" ]; then
        printf "  Docker:  ready\n"
    else
        printf "  Docker:  not available (WASM-only mode)\n"
    fi

    printf "\n"
    printf "${BOLD}To start the agent:${RESET}\n"
    printf "\n"

    case ":$PATH:" in
        *":$BIN_DIR:"*)
            printf "  archipelag-agent --agent --config %s\n" "$CONFIG_DIR/config.toml"
            ;;
        *)
            printf "  source %s\n" "$RC_FILE"
            printf "  archipelag-agent --agent --config %s\n" "$CONFIG_DIR/config.toml"
            ;;
    esac

    printf "\n"
    printf "  Docs: https://github.com/%s\n" "$REPO"
    printf "\n"
}

# Main
main() {
    printf "\n${BOLD}Archipelag.io Node Agent Installer${RESET}\n\n"

    detect_platform
    info "Detected platform: $PLATFORM"

    get_latest_version
    download_binary
    verify_binary
    setup_docker
    create_config
    setup_path
    print_success
}

main
