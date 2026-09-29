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
PORT="${ROBODOJO_PORT:-7891}"          # base port; proc i listens on PORT+i
PROCS="${ROBODOJO_PROCS:-1}"           # number of independent policy-server processes
TASK="${ROBODOJO_TASK:-cover_blocks}"
BIND_HOST="${ROBODOJO_BIND_HOST:-127.0.0.1}"
GPU="${ROBODOJO_GPU:-0}"
BASE_PY="${OPENDM_PYTHON:-/root/miniconda3/envs/opendm/bin/python}"

# One log file per process, plus a pid/port table used by status/stop.
LOG_DIR="${ROBODOJO_LOG_DIR:-/tmp/robodojo_policy_server}"
PORTS_FILE="$LOG_DIR/ports"
READY_TIMEOUT="${ROBODOJO_READY_TIMEOUT:-420}"

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
    --port N               Base TCP port (default 7891); process i listens on N+i
    --procs K              Independent server processes (default 1)
    --bind-host HOST       Bind address (default 127.0.0.1; keep loopback when tunnelling)
    --task NAME            Task name recorded by the server (default cover_blocks)
    --gpu ID               CUDA_VISIBLE_DEVICES for the policy server (default 0)

    Each process loads its own copy of the weights, so VRAM scales with --procs
    (~23 GB each for DM0.5). Two processes also lift the single-process asyncio
    model lock, which is what caps one process at ~1 inference/s.

  status | stop
    --port N               Base port used when the servers were started (default 7891)
    --procs K              Expected process count, for reporting (default 1)
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

listening() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -lnt 2>/dev/null | grep -q ":$port "
    else
        timeout 1 bash -c "</dev/tcp/$BIND_HOST/$port" 2>/dev/null
    fi
}

# Readiness is judged by the listening socket, not by stdout: the server's
# progress bar is block-buffered and useless as a signal.
wait_for_port() {
    local port="$1" deadline=$((SECONDS + READY_TIMEOUT))
    while [ "$SECONDS" -lt "$deadline" ]; do
        listening "$port" && return 0
        sleep 3
    done
    return 1
}

do_start() {
    require_staged
    case "$PROCS" in ''|*[!0-9]*) die "--procs must be a positive integer, got: $PROCS" ;; esac
    [ "$PROCS" -ge 1 ] || die "--procs must be >= 1"

    local last_port=$((PORT + PROCS - 1))
    info "serving $POLICY_NAME on $BIND_HOST:$PORT..$last_port  (procs=$PROCS, task=$TASK, gpu=$GPU)"
    info "each process loads its own copy of $CKPT (~23 GB VRAM each)"

    mkdir -p "$LOG_DIR"
    : > "$PORTS_FILE"

    export MODEL_PATH="$CKPT"
    export NORM_STATS_PATH="$CKPT/norm_stats.json"   # a FILE, not a directory
    export PYTHONPATH="$XPL_DIR${PYTHONPATH:+:$PYTHONPATH}"
    # A stale proxy in the container image breaks model loading; clear it.
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

    local i port log
    local failed=0
    for (( i=0; i<PROCS; i++ )); do
        port=$((PORT + i))
        log="$LOG_DIR/server_${port}.log"
        echo "$port" >> "$PORTS_FILE"
        # Run from the policy dir: setup_policy_server.py imports client_server.*
        # from beside itself, and the adapter pulls exp/opendm from the staged
        # tree. setsid + </dev/null fully detaches the process, so a closing SSH
        # session cannot take the server with it (a foreground start used to be
        # killed by the caller's timeout, losing all of its output).
        setsid bash -c "
            cd '$POLICY_DIR' || exit 1
            exec env PYTHONWARNINGS=ignore::UserWarning CUDA_VISIBLE_DEVICES='$GPU' \
                '$VENV/bin/python' '$XPL_ROOT/setup_policy_server.py' \
                    --config_path '$POLICY_DIR/deploy.yml' \
                    --overrides \
                        port='$port' host='$BIND_HOST' bench_name=RoboDojo \
                        task_name='$TASK' ckpt_name=external env_cfg_type='$ENV_CFG' \
                        seed=0 policy_name='$POLICY_NAME' action_type='$ACTION_TYPE' \
                        action_steps='$ACTION_STEPS' model_path='$CKPT' \
                        norm_stats_path='$CKPT/norm_stats.json'
        " > "$log" 2>&1 < /dev/null &
        info "proc $((i + 1))/$PROCS  port=$port  log=$log"

        # Load strictly one at a time. A single process peaks around 60 GB of
        # host RAM while materialising the F32 weights, so bringing K of them
        # up together OOMs the container (119 GB cgroup on this box) even
        # though the GPUs would cope.
        if wait_for_port "$port"; then
            info "port $port ready"
        else
            warn "port $port did not come up -- see $log"
            failed=1
            break
        fi
    done

    command -v nvidia-smi >/dev/null 2>&1 && \
        echo "gpu    : $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1)"
    echo
    echo "client endpoint:  --policy-host $BIND_HOST --policy-port $(paste -sd, "$PORTS_FILE")"
    [ "$failed" -eq 0 ] || die "one or more policy servers failed to start"
}

# Matches the server process without matching this script's own command line.
SERVER_PATTERN='setup_policy_serve[r].py'

do_status() {
    local pids count pid port
    # `|| true` keeps `set -e` happy when pgrep matches nothing.
    pids="$(pgrep -f "$SERVER_PATTERN" || true)"
    count="$(printf '%s' "$pids" | grep -c . || true)"
    echo "procs  : $count running (expected $PROCS on ports $PORT..$((PORT + PROCS - 1)))"
    if [ "$count" -gt 0 ]; then
        while read -r pid; do
            [ -n "$pid" ] || continue
            printf '         pid %-8s %s\n' "$pid" \
                "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -o 'port=[0-9]*' | head -1)"
        done <<< "$pids"
    fi
    for (( port=PORT; port<PORT+PROCS; port++ )); do
        if listening "$port"; then
            echo "port   : $port listening"
        else
            echo "port   : $port NOT listening"
        fi
    done
    command -v nvidia-smi >/dev/null 2>&1 && \
        echo "gpu    : $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1)"
    echo "staged : $([ -d "$POLICY_DIR" ] && echo yes || echo no)  |  venv: $([ -x "$VENV/bin/python" ] && echo yes || echo no)"
}

do_stop() {
    local pids count
    pids="$(pgrep -f "$SERVER_PATTERN" || true)"
    if [ -n "$pids" ]; then
        count="$(printf '%s' "$pids" | grep -c . || true)"
        info "stopping $count policy server process(es)"
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
                    --procs)     PROCS="$2"; shift 2 ;;
                    --bind-host) BIND_HOST="$2"; shift 2 ;;
                    --task)      TASK="$2"; shift 2 ;;
                    --gpu)       GPU="$2"; shift 2 ;;
                    -h|--help)   usage; exit 0 ;;
                    *) die "unknown option for start: $1" ;;
                esac
            done
            do_start
            ;;
        status|stop)
            # --port/--procs only shape the report; a single pgrep/pkill covers
            # every process this script ever started.
            while [ "$#" -gt 0 ]; do
                case "$1" in
                    --port)  PORT="$2"; shift 2 ;;
                    --procs) PROCS="$2"; shift 2 ;;
                    -h|--help) usage; exit 0 ;;
                    *) die "unknown option for $cmd: $1" ;;
                esac
            done
            if [ "$cmd" = "status" ]; then do_status; else do_stop; fi
            ;;
        -h|--help|help) usage ;;
        *) usage; exit 2 ;;
    esac
}

main "$@"
