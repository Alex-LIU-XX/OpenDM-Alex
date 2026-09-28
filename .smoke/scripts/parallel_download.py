#!/usr/bin/env python3
"""Dexmal/DM05 model.safetensors 多线程分片下载器。

背景：hf-mirror 的单连接吞吐只有 ~3-5 MB/s，11.66 GB 的主权重按 hf download
默认路径需要 60+ 分钟；实测 6-8 条 HTTP Range 连接可跑满 ~50-80 MB/s。

策略：
  1. 把文件切成 CHUNK 大小的分片，用 ThreadPoolExecutor 并发下载（每个分片一个 curl）。
  2. 每个分片先写 part.NNNN.tmp，长度校验通过后再落成 part.NNNN，避免服务端忽略
     Range 返回 200 时把整份文件写进分片。
  3. 全部完成后按序拼接，计算 sha256 与 HF 上登记的 LFS oid 比对。
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

URL = "https://hf-mirror.com/Dexmal/DM05/resolve/main/model.safetensors"
DEST = "/root/workspace/opendm/checkpoints/DM05/model.safetensors"
PARTS_DIR = "/root/workspace/opendm/checkpoints/.parts"

TOTAL = 11_658_431_136
SHA256 = "b7da77f516ebd0c0c68faed2f0cfe1c68985b7b874e49dda4292d8d286de20dd"
CHUNK = 256 * 1024 * 1024
WORKERS = 8
MAX_ATTEMPTS = 15

_lock = threading.RLock()
_done_bytes = 0
_t0 = time.time()


def log(msg: str) -> None:
    with _lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_chunks() -> list[tuple[int, int, int]]:
    chunks = []
    off = 0
    idx = 0
    while off < TOTAL:
        end = min(off + CHUNK, TOTAL) - 1
        chunks.append((idx, off, end))
        off = end + 1
        idx += 1
    return chunks


def fetch_one(chunk: tuple[int, int, int]) -> tuple[int, bool]:
    global _done_bytes
    idx, start, end = chunk
    want = end - start + 1
    final = os.path.join(PARTS_DIR, f"part.{idx:04d}")
    tmp = final + ".tmp"

    if os.path.exists(final) and os.path.getsize(final) == want:
        with _lock:
            _done_bytes += want
        return idx, True

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if os.path.exists(tmp):
            os.remove(tmp)
        cmd = [
            "curl", "-s", "--noproxy", "*", "-L",
            "--retry", "6", "--retry-delay", "3", "--retry-all-errors",
            "--connect-timeout", "30",
            "--speed-time", "60", "--speed-limit", "20000",
            "-r", f"{start}-{end}",
            "-o", tmp, URL,
        ]
        try:
            rc = subprocess.run(cmd, timeout=900).returncode
        except subprocess.TimeoutExpired:
            rc = -1

        got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if rc == 0 and got == want:
            os.replace(tmp, final)
            # 注意：log() 自己会拿 _lock，必须在锁外调用（Lock 不可重入会死锁）
            with _lock:
                _done_bytes += want
                pct = 100.0 * _done_bytes / TOTAL
                mbps = _done_bytes / 1048576 / max(time.time() - _t0, 1e-6)
            log(f"chunk {idx:>3}/{len(CHUNKS)-1} ok  {pct:5.1f}%  {mbps:6.1f} MB/s avg")
            return idx, True

        if attempt % 3 == 0 or attempt == 1:
            log(f"chunk {idx:>3} attempt {attempt} failed (rc={rc}, {got}/{want} B), retrying")

    log(f"chunk {idx:>3} GAVE UP")
    return idx, False


def concat() -> None:
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    log(f"concatenating {len(CHUNKS)} parts -> {DEST}")
    with open(DEST, "wb") as out:
        for idx, _, _ in CHUNKS:
            part = os.path.join(PARTS_DIR, f"part.{idx:04d}")
            with open(part, "rb") as fh:
                while True:
                    buf = fh.read(16 * 1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(16 * 1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def progress_loop(stop: threading.Event) -> None:
    while not stop.wait(60):
        with _lock:
            done = _done_bytes
        el = time.time() - _t0
        log(f"progress {100.0*done/TOTAL:5.1f}%  {done/2**30:.2f}/{TOTAL/2**30:.2f} GiB  "
            f"{done/1048576/max(el,1e-6):5.1f} MB/s  eta {max(TOTAL-done,0)/max(done/max(el,1e-6),1e-6)/60:.1f} min")


def main() -> int:
    global CHUNKS
    CHUNKS = build_chunks()
    os.makedirs(PARTS_DIR, exist_ok=True)
    # 清掉上次中断留下的半成品分片
    for name in os.listdir(PARTS_DIR):
        if name.endswith(".tmp"):
            os.remove(os.path.join(PARTS_DIR, name))
    log(f"total={TOTAL} B, chunks={len(CHUNKS)}, chunk_size={CHUNK}, workers={WORKERS}")
    log(f"url={URL}")

    stop = threading.Event()
    reporter = threading.Thread(target=progress_loop, args=(stop,), daemon=True)
    reporter.start()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(fetch_one, CHUNKS))
    stop.set()

    failed = [idx for idx, ok in results if not ok]
    if failed:
        log(f"FAILED chunks: {failed}")
        return 1

    el = time.time() - _t0
    log(f"all chunks ok in {el/60:.1f} min ({TOTAL/1048576/el:.1f} MB/s avg)")

    if os.path.exists(DEST):
        os.remove(DEST)
    concat()

    size = os.path.getsize(DEST)
    if size != TOTAL:
        log(f"SIZE MISMATCH: got {size}, want {TOTAL}")
        return 1

    log("verifying sha256 ...")
    got = sha256_of(DEST)
    if got != SHA256:
        log(f"SHA256 MISMATCH:\n  got  {got}\n  want {SHA256}")
        return 1

    log(f"SHA256 OK: {got}")
    log(f"size OK: {size} B ({size/2**30:.2f} GiB)")
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
