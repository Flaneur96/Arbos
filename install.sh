#!/usr/bin/env bash
#
# Agent — one-command install
#
# Usage:
#   ./install.sh
#   curl -fsSL <url>/install.sh | bash   (interactive)
#

set -e
set -o pipefail

# ── Colors ───────────────────────────────────────────────────────────────────

if [ -t 1 ] && [ "${TERM:-dumb}" != "dumb" ]; then
    GREEN=$'\033[0;32m' RED=$'\033[0;31m' CYAN=$'\033[0;36m'
    BOLD=$'\033[1m' DIM=$'\033[2m' NC=$'\033[0m'
else
    GREEN='' RED='' CYAN='' BOLD='' DIM='' NC=''
fi

ok()  { printf "  ${GREEN}+${NC} %s\n" "$1"; }
err() { printf "  ${RED}x${NC} %s\n" "$1"; }
die() { err "$1"; exit 1; }

command_exists() { command -v "$1" >/dev/null 2>&1; }

# ── Spinner ──────────────────────────────────────────────────────────────────

spin() {
    local pid=$1 msg="$2" i=0 chars='|/-\'
    printf "\033[?25l" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null; do
        printf "\r  ${CYAN}%s${NC} %s" "${chars:$((i%4)):1}" "$msg"
        sleep 0.1 2>/dev/null || sleep 1
        i=$((i+1))
    done
    printf "\033[?25h" 2>/dev/null || true
    wait "$pid" 2>/dev/null; local code=$?
    if [ $code -eq 0 ]; then
        printf "\r  ${GREEN}+${NC} %s\n" "$msg"
    else
        printf "\r  ${RED}x${NC} %s\n" "$msg"
    fi
    return $code
}

run() {
    local msg="$1"; shift
    local tmp_out=$(mktemp) tmp_err=$(mktemp)
    "$@" >"$tmp_out" 2>"$tmp_err" &
    local pid=$!
    if ! spin $pid "$msg"; then
        if [ -s "$tmp_err" ]; then
            printf "\n    ${RED}${BOLD}stderr:${NC}\n"
            while IFS= read -r l; do printf "    ${DIM}%s${NC}\n" "$l"; done < "$tmp_err"
        elif [ -s "$tmp_out" ]; then
            printf "\n    ${RED}${BOLD}output:${NC}\n"
            tail -20 "$tmp_out" | while IFS= read -r l; do printf "    ${DIM}%s${NC}\n" "$l"; done
        fi
        printf "\n"
        rm -f "$tmp_out" "$tmp_err"
        return 1
    fi
    rm -f "$tmp_out" "$tmp_err"
}

# ── Detect context ───────────────────────────────────────────────────────────

INSTALL_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
[ -z "$INSTALL_DIR" ] && INSTALL_DIR="$(pwd)"

HAS_TTY=false
if [ -t 0 ] || { [ -e /dev/tty ] && (echo >/dev/tty) 2>/dev/null; }; then
    HAS_TTY=true
fi

# ── Banner ───────────────────────────────────────────────────────────────────

printf "\n${CYAN}${BOLD}"
printf "      _                    _   \n"
printf "     / \\   __ _  ___ _ __ | |_ \n"
printf "    / _ \\ / _\` |/ _ \\ '_ \\| __|\n"
printf "   / ___ \\ (_| |  __/ | | | |_ \n"
printf "  /_/   \\_\\__, |\\___|_| |_|\\__|\n"
printf "          |___/                \n"
printf "${NC}\n"

# ── 1. Detect package manager ────────────────────────────────────────────────

pkg_install() {
    if command_exists apt-get; then
        sudo apt-get update -qq && sudo apt-get install -y -qq "$@"
    elif command_exists dnf; then
        sudo dnf install -y -q "$@"
    elif command_exists yum; then
        sudo yum install -y -q "$@"
    elif command_exists pacman; then
        sudo pacman -S --noconfirm --needed "$@"
    elif command_exists brew; then
        brew install "$@"
    else
        die "No supported package manager found (apt/dnf/yum/pacman/brew)"
    fi
}

# ── 2. Install prerequisites ────────────────────────────────────────────────

printf "  ${BOLD}Installing prerequisites${NC}\n\n"

for cmd in git python3 curl tmux vim; do
    if command_exists "$cmd"; then
        ok "$cmd"
    else
        run "Installing $cmd" pkg_install "$cmd"
        command_exists "$cmd" || die "Failed to install $cmd"
    fi
done

printf "\n"

# ── 3. Install tooling ──────────────────────────────────────────────────────

printf "  ${BOLD}Installing tooling${NC}\n\n"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# uv
if command_exists uv; then
    ok "uv already installed"
else
    run "Installing uv" bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command_exists uv || die "uv install failed"
fi

# Cursor
if command_exists cursor; then
    ok "Cursor already installed"
else
    CURSOR_URL="https://www.cursor.com/api/download?platform=linux-x64&releaseTrack=stable"
    CURSOR_APPIMAGE="$HOME/.local/bin/cursor.AppImage"
    mkdir -p "$HOME/.local/bin"
    run "Installing Cursor" bash -c "curl -L '$CURSOR_URL' -o '$CURSOR_APPIMAGE' && chmod +x '$CURSOR_APPIMAGE' && ln -sf '$CURSOR_APPIMAGE' '$HOME/.local/bin/cursor'"
    command_exists cursor || die "Cursor install failed"
fi

# PATH persistence
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    SHELL_RC="$HOME/.bashrc"
    [[ -n "${ZSH_VERSION:-}" ]] && SHELL_RC="$HOME/.zshrc"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
    export PATH="$HOME/.local/bin:$PATH"
    ok "Added ~/.local/bin to PATH"
fi

printf "\n"

# ── 4. Python environment ───────────────────────────────────────────────────

printf "  ${BOLD}Setting up project${NC}\n\n"

cd "$INSTALL_DIR"

if [ ! -d ".venv" ]; then
    run "Creating Python environment" uv venv .venv
else
    ok "Python environment exists"
fi

source .venv/bin/activate
run "Installing dependencies" uv pip install -e .

mkdir -p history scratch

printf "\n"

# ── 5. Edit PROMPT.md ───────────────────────────────────────────────────────

printf "  ${BOLD}Prompt${NC}\n\n"

printf "  ${CYAN}PROMPT.md${NC} is the system prompt fed to the agent at every step.\n"
printf "  It tells the agent who it is, where to find its goal, how to\n"
printf "  store history, and any persistent hints you want to pass along.\n\n"
printf "  ${DIM}You're about to open it in vim — edit it to your liking,${NC}\n"
printf "  ${DIM}then save and quit (:wq) to continue the install.${NC}\n\n"

if [ "$HAS_TTY" = true ]; then
    printf "  ${DIM}Press any key to open PROMPT.md in vim...${NC}"
    read -rsn1 </dev/tty 2>/dev/null || true
    printf "\n\n"

    vim "$INSTALL_DIR/PROMPT.md" </dev/tty >/dev/tty

    ok "PROMPT.md saved"
else
    [ -f "$INSTALL_DIR/PROMPT.md" ] || die "No TTY and no PROMPT.md — cannot configure"
    ok "PROMPT.md exists (no TTY, skipping editor)"
fi

printf "\n"

# ── 6. Collect .env ──────────────────────────────────────────────────────────

printf "  ${BOLD}Configuration${NC}\n\n"

if [ "$HAS_TTY" = true ]; then
    printf "  ${DIM}Paste your environment variables (KEY=VALUE, one per line)${NC}\n"
    printf "  ${DIM}Press Enter on an empty line when done${NC}\n\n"

    ENV_CONTENT=""
    while true; do
        printf "  ${CYAN}>${NC} "
        IFS= read -r line </dev/tty 2>/dev/null || break
        [ -z "$line" ] && break
        ENV_CONTENT+="$line"$'\n'
    done

    if [ -n "$ENV_CONTENT" ]; then
        printf '%s' "$ENV_CONTENT" > "$INSTALL_DIR/.env"
        ok ".env saved"
    else
        if [ -f "$INSTALL_DIR/.env" ]; then
            ok ".env unchanged (kept existing)"
        else
            err "No .env content provided"
        fi
    fi
else
    [ -f "$INSTALL_DIR/.env" ] || die "No TTY and no .env — cannot configure"
    ok ".env exists"
fi

printf "\n"

# ── 7. Start agent in tmux ───────────────────────────────────────────────────

SESSION="agent"

printf "  ${BOLD}Starting agent${NC}\n\n"

# Kill existing session if present
if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    ok "Killed previous session"
fi

tmux new-session -d -s "$SESSION" -c "$INSTALL_DIR" \
    "$INSTALL_DIR/.venv/bin/python3 $INSTALL_DIR/agent.py"

sleep 1
if tmux has-session -t "$SESSION" 2>/dev/null; then
    ok "Agent running in tmux session '${SESSION}'"
else
    die "Agent failed to start"
fi

# ── Done ─────────────────────────────────────────────────────────────────────

printf "\n"
printf "  ${GREEN}${BOLD}Agent is live${NC}\n"
printf "\n"
printf "  ${DIM}attach${NC}   tmux attach -t ${SESSION}\n"
printf "  ${DIM}logs${NC}     tmux attach -t ${SESSION} ${DIM}(Ctrl+B D to detach)${NC}\n"
printf "  ${DIM}kill${NC}     tmux kill-session -t ${SESSION}\n"
printf "\n"
