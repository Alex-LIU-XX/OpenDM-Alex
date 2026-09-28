#!/usr/bin/env bash
# RoboDojo split-eval policy server (server / policy side).
#
# Serves the RoboDojo-tuned DM0.5 checkpoint over the XPolicyLab WebSocket
# policy protocol, so that a RoboDojo simulator client running on another host
# can drive inference against it.
#
# Typical use on the policy host:
#   bash script/robodojo_policy_server.sh setup    # once: stage code + env
#   bash script/robodojo_policy_server.sh start    # serve on 127.0.0.1:7891
#   bash script/robodojo_policy_server.sh status
#   bash script/robodojo_policy_server.sh stop
#
# The client host then tunnels the port and runs `robodojo.sh client`.
# Full walkthrough: docs/Alex/robodojo复现/分体推理使用文档.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --------------------------------------------------------------------------
# Defaults (override with flags or environment variables)
# --------------------------------------------------------------------------
WORKDIR="${ROBODOJO_SPLIT_WORKDIR:-/root/workspace/Alex}"
XPL_DIR="$WORKDIR/xpl"                    # holds XPolicyLab/ + env_cfg/
VENV="$WORKDIR/policy_venv"
ROBODOJO_REPO="${ROBODOJO_REPO:-/root/workspace/RoboDojo}"   # source of XPolicyLab + env_cfg
CKPT="${ROBODOJO_CKPT:-/mnt/cfs/opendm/checkpoints/DM05-MEM-Robodojo-Sim}"
PORT="${ROBODOJO_PORT:-7891}"
TASK="${ROBODOJO_TASK:-cover_blocks}"
BIND_HOST="${ROBODOJO_BIND_HOST:-127.0.0.1}"
GPU="${ROBODOJO_GPU:-0}"
BASE_PY="${OPENDM_PYTHON:-/root/miniconda3/envs/opendm/bin/python}"

# Inference contract of the RoboDojo leaderboard model - do not change casually.
POLICY_NAME="OpenDM"
ENV_CFG="arx_x5"
ACTION_TYPE="joint"
ACTION_STEPS=25

XPL_ROOT="$XPL_DIR/XPolicyLab"
POLICY_DIR="$XPL_ROOT/policy/$POLICY_NAME"

# XPolicyLab dependencies missing from the shared conda env are installed into
# an overlay venv (--system-site-packages) so the shared env stays untouched.
VENV_DEPS=(h5py websockets msgpack msgpack-numpy)
# policy/Pi_05 alone is ~9.4G of checkpoints we do not need for this contract.
COPY_EXCLUDES=(--exclude=.git --exclude=__pycache__ --exclude='XPolicyLab/policy/Pi_05')

FORCE="false"

info()  { printf '\033[1;32m>>> %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m>>> %s\033[0m\n' "$*"; }
die()   { printf '\033[1;31m[ERROR] %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  setup
    --robodojo-repo PATH   RoboDojo checkout used as source for XPolicyLab + env_cfg
                           (default: /root/workspace/RoboDojo)
    --workdir PATH         Where to stage code + venv (default: /root/workspace/Alex)
    --force                Re-copy code even if already staged

  start
    --ckpt PATH            Checkpoint dir (default: DM05-MEM-Robodojo-Sim on CFS)
    --port N               TCP port (default 7891)
    --bind-host HOST       Bind address (default 127.0.0.1; keep loopback when tunnelling)
    --task NAME            Task name recorded by the server (default cover_blocks)
    --gpu ID               CUDA_VISIBLE_DEVICES for the policy server (default 0)

  status | stop
    no options
EOF
}

# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
do_setup() {
    info "setup: staging RoboDojo policy code under $XPL_DIR"

    [ -d "$ROBODOJO_REPO/XPolicyLab" ] || die "no XPolicyLab under $ROBODOJO_REPO (pass --robodojo-repo)"
    [ -d "$ROBODOJO_REPO/env_cfg" ]    || die "no env_cfg under $ROBODOJO_REPO (pass --robodojo-repo)"
    [ -x "$BASE_PY" ]                  || die "base python not found: $BASE_PY (set OPENDM_PYTHON)"

    mkdir -p "$XPL_DIR"

    if [ -e "$XPL_ROOT" ] && [ "$FORCE" != "true" ]; then
        info "XPolicyLab already staged, skipping copy (use --force to refresh)"
    else
        info "copying XPolicyLab (excluding .git, __pycache__, policy/Pi_05)"
        rm -rf "$XPL_ROOT"
        tar czf - -C "$ROBODOJO_REPO" "${COPY_EXCLUDES[@]}" XPolicyLab \
            | tar xzf - -C "$XPL_DIR"
    fi

    # The adapter resolves robot action dims from <repo>/env_cfg/<stem>.yml,
    # i.e. a sibling of XPolicyLab - it must match the client's env_cfg.
    info "copying env_cfg"
    rm -rf "$XPL_DIR/env_cfg"
    tar czf - -C "$ROBODOJO_REPO" env_cfg | tar xzf - -C "$XPL_DIR"

    info "checking staged files"
    local missing=0 f
    for f in "policy/$POLICY_NAME/setup_eval_policy_server.sh" \
             "policy/$POLICY_NAME/deploy.yml" \
             "policy/$POLICY_NAME/model.py" \
             "client_server/ws/model_server.py" \
             "utils/get_free_port.sh" \
             "setup_policy_server.py"; do
        [ -e "$XPL_ROOT/$f" ] || { warn "missing: $f"; missing=1; }
    done
    [ -e "$XPL_DIR/env_cfg/$ENV_CFG.yml" ] || { warn "missing: env_cfg/$ENV_CFG.yml"; missing=1; }
    [ "$missing" -eq 0 ] || die "staged tree is incomplete"

    info "preparing overlay venv at $VENV"
    [ -x "$VENV/bin/python" ] || "$BASE_PY" -m venv --system-site-packages "$VENV"
    (
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
        "$VENV/bin/pip" install -q --timeout 30 "${VENV_DEPS[@]}"
    )

    info "verifying imports"
    "$VENV/bin/python" - <<'PY'
import importlib.util as iu
import sys
missing = [m for m in ("h5py", "websockets", "msgpack", "msgpack_numpy", "torch")
           if iu.find_spec(m) is None]
if missing:
    sys.exit("missing modules: " + ", ".join(missing))
import torch
print(f"  torch {torch.__version__} | cuda_available={torch.cuda.is_available()}")
PY

    [ -e "$CKPT/model.safetensors" ] || warn "checkpoint not found: $CKPT (pass --ckpt to start)"
    [ -e "$CKPT/norm_stats.json" ]   || warn "norm_stats.json not found in $CKPT"

    info "setup done"
    echo
    echo "  code     : $XPL_ROOT"
    echo "  env_cfg  : $XPL_DIR/env_cfg"
    echo "  venv     : $VENV"
    echo "  ckpt     : $CKPT"
    echo
    echo "Next:  bash $0 start"
}

# --------------------------------------------------------------------------
# start / status / stop
# --------------------------------------------------------------------------
require_staged() {
    [ -d "$POLICY_DIR" ]    || die "policy dir not staged: $POLICY_DIR (run: bash $0 setup)"
    [ -x "$VENV/bin/python" ] || die "venv missing: $VENV (run: bash $0 setup)"
    [ -e "$CKPT/model.safetensors" ] || die "checkpoint not found: $CKPT"
    [ -e "$CKPT/norm_stats.json" ]   || die "norm_stats.json not found in $CKPT"
}

do_start() {
    require_staged
    info "serving $POLICY_NAME on $BIND_HOST:$PORT  (task=$TASK, gpu=$GPU)"
    info "loading $CKPT - first run takes ~2-3 min, later runs ~20-40s"

    export MODEL_PATH="$CKPT"
    export NORM_STATS_PATH="$CKPT/norm_stats.json"   # a FILE, not a directory
    export PYTHONPATH="$XPL_DIR${PYTHONPATH:+:$PYTHONPATH}"
    # A stale proxy in the container image breaks model loading; clear it.
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

    # setup_policy_server.py imports client_server.* from its own directory and
    # the adapter pulls exp/opendm from the staged tree.
    cd "$POLICY_DIR"
    exec env PYTHONWARNINGS=ignore::UserWarning CUDA_VISIBLE_DEVICES="$GPU" \
        "$VENV/bin/python" "$XPL_ROOT/setup_policy_server.py" \
            --config_path "$POLICY_DIR/deploy.yml" \
            --overrides \
                port="$PORT" \
                host="$BIND_HOST" \
                bench_name=RoboDojo \
                task_name="$TASK" \
                ckpt_name=external \
                env_cfg_type="$ENV_CFG" \
                seed=0 \
                policy_name="$POLICY_NAME" \
                action_type="$ACTION_TYPE" \
                action_steps="$ACTION_STEPS" \
                model_path="$CKPT" \
                norm_stats_path="$CKPT/norm_stats.json"
}

# Matches the server process without matching this script's own command line.
SERVER_PATTERN='setup_policy_serve[r].py'

do_status() {
    local pid
    pid="$(pgrep -f "$SERVER_PATTERN" | head -1 || true)"
    if [ -n "$pid" ]; then
        echo "server : running (pid $pid)"
        tr '\0' ' ' < "/proc/$pid/cmdline" | sed 's/--overrides.*/--overrides .../' | fold -w 120 | sed 's/^/         /'
        echo
    else
        echo "server : not running"
    fi
    if command -v ss >/dev/null 2>&1 && ss -lntp 2>/dev/null | grep -q ":$PORT"; then
        echo "port   : $PORT listening"
    else
        echo "port   : $PORT not listening"
    fi
    command -v nvidia-smi >/dev/null 2>&1 && \
        echo "gpu    : $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1)"
    echo "staged : $([ -d "$POLICY_DIR" ] && echo yes || echo no)  |  venv: $([ -x "$VENV/bin/python" ] && echo yes || echo no)"
}

do_stop() {
    if pgrep -f "$SERVER_PATTERN" >/dev/null; then
        pkill -f "$SERVER_PATTERN" || true
        sleep 3
        pgrep -f "$SERVER_PATTERN" >/dev/null && die "server still running" || info "server stopped"
    else
        info "server was not running"
    fi
    command -v nvidia-smi >/dev/null 2>&1 && \
        echo "gpu    : $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1)"
}

# --------------------------------------------------------------------------
main() {
    [ "$#" -ge 1 ] || { usage; exit 2; }
    local cmd="$1"; shift

    case "$cmd" in
        setup)
            while [ "$#" -gt 0 ]; do
                case "$1" in
                    --robodojo-repo) ROBODOJO_REPO="$2"; shift 2 ;;
                    --workdir)       WORKDIR="$2"
                                     XPL_DIR="$WORKDIR/xpl"; VENV="$WORKDIR/policy_venv"
                                     XPL_ROOT="$XPL_DIR/XPolicyLab"; POLICY_DIR="$XPL_ROOT/policy/$POLICY_NAME"
                                     shift 2 ;;
                    --force)         FORCE="true"; shift ;;
                    -h|--help)       usage; exit 0 ;;
                    *) die "unknown option for setup: $1" ;;
                esac
            done
            do_setup
            ;;
        start)
            while [ "$#" -gt 0 ]; do
                case "$1" in
                    --ckpt)      CKPT="$2"; shift 2 ;;
                    --port)      PORT="$2"; shift 2 ;;
                    --bind-host) BIND_HOST="$2"; shift 2 ;;
                    --task)      TASK="$2"; shift 2 ;;
                    --gpu)       GPU="$2"; shift 2 ;;
                    -h|--help)   usage; exit 0 ;;
                    *) die "unknown option for start: $1" ;;
                esac
            done
            do_start
            ;;
        status) do_status ;;
        stop)   do_stop ;;
        -h|--help|help) usage ;;
        *) usage; exit 2 ;;
    esac
}

main "$@"
