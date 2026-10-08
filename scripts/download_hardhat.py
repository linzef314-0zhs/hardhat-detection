"""Parallel resumable downloader for the Hard Hat Detection dataset (Kaggle public endpoint).

Campus-network friendly: HTTP Range requests, per-part resume, retries.
Writes progress to logs/download.log so the agent can poll.
"""
import os
import sys
import time
import threading
import urllib.request

URL = "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/hard-hat-detection"
OUT = sys.argv[1] if len(sys.argv) > 1 else "hardhat.zip"
LOG = sys.argv[2] if len(sys.argv) > 2 else "download.log"
N_THREADS = 8
CHUNK_MIN = 4 * 1024 * 1024

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def resolve(url):
    # Kaggle endpoint does not support HEAD; probe with a 1-byte Range GET.
    req = urllib.request.Request(url)
    req.add_header("Range", "bytes=0-0")
    with urllib.request.urlopen(req, timeout=30) as r:
        cr = r.headers.get("Content-Range", "")
        total = int(cr.split("/")[-1])
        return r.geturl(), total

def download_range(final_url, start, end, part_path, idx, progress, lock):
    done = os.path.getsize(part_path) if os.path.exists(part_path) else 0
    target = end - start + 1
    retries = 0
    while done < target:
        try:
            req = urllib.request.Request(final_url)
            req.add_header("Range", f"bytes={start + done}-{end}")
            with urllib.request.urlopen(req, timeout=60) as r, open(part_path, "ab") as f:
                while True:
                    buf = r.read(1024 * 256)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    with lock:
                        progress[0] += len(buf)
        except Exception as e:
            retries += 1
            if retries > 50:
                raise RuntimeError(f"part {idx} failed after 50 retries: {e}")
            time.sleep(min(retries * 2, 30))
    log(f"part {idx} done ({target} bytes)")

def main():
    final_url, total = resolve(URL)
    log(f"resolved: {final_url[:80]}... total={total/1e6:.1f} MB")
    if os.path.exists(OUT) and os.path.getsize(OUT) == total:
        log("already complete")
        return
    boundaries = []
    step = max(total // N_THREADS, CHUNK_MIN)
    s = 0
    while s < total:
        e = min(s + step - 1, total - 1)
        boundaries.append((s, e))
        s = e + 1
    progress = [0]
    lock = threading.Lock()
    threads = []
    for i, (s, e) in enumerate(boundaries):
        part = f"{OUT}.part{i}"
        t = threading.Thread(target=download_range, args=(final_url, s, e, part, i, progress, lock))
        t.daemon = True
        t.start()
        threads.append(t)
    last = 0
    t0 = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        cur = progress[0]
        speed = (cur - last) / 15 / 1e6
        last = cur
        log(f"progress {cur/1e6:.1f}/{total/1e6:.1f} MB ({100*cur/total:.1f}%) speed={speed:.2f} MB/s elapsed={time.time()-t0:.0f}s")
    for t in threads:
        t.join()
    # 校验按分块实际大小（兼容断点续传，progress 只计数本次新增字节）
    for i, (s, e) in enumerate(boundaries):
        part = f"{OUT}.part{i}"
        want = e - s + 1
        got = os.path.getsize(part) if os.path.exists(part) else 0
        if got != want:
            raise RuntimeError(f"part {i} incomplete: {got}/{want}")
    with open(OUT, "wb") as out:
        for i in range(len(boundaries)):
            part = f"{OUT}.part{i}"
            with open(part, "rb") as f:
                while True:
                    buf = f.read(1024 * 1024 * 8)
                    if not buf:
                        break
                    out.write(buf)
            os.remove(part)
    assert os.path.getsize(OUT) == total, "size mismatch"
    log(f"DONE: {OUT} ({total/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
