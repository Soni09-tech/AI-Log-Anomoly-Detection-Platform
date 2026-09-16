"""
Synthetic log generator (HDFS-style).

Real datasets to use instead once you have internet access:
  - Loghub HDFS: https://github.com/logpanpan/loghub  (or https://github.com/logpub/loghub)
  - Loghub BGL, Hadoop, Linux, etc.
These come with ground-truth anomaly labels per block/session.

This script generates a similar structure so you can build and test the
full pipeline offline, then swap in the real dataset later (the parser
and feature code don't care where the logs came from).

Each "session" = one block_id. A session is a sequence of log events.
Most sessions follow a "normal" template sequence. A fraction are
corrupted with anomalous events (missing steps, error events, retries)
and labeled anomaly=1.
"""

import random
import string
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" if (ROOT / "data").exists() else ROOT

random.seed(42)

NORMAL_EVENTS = [
    "Receiving block {blk} src: /{ip}:{port} dest: /{ip2}:{port2}",
    "PacketResponder {n} for block {blk} terminating",
    "Received block {blk} of size {size} from /{ip}",
    "BLOCK* NameSystem.addStoredBlock: blockMap updated: {ip}:{port} is added to {blk} size {size}",
    "Verification succeeded for {blk}",
    "BLOCK* ask {ip}:{port} to delete {blk}",
]

ANOMALY_EVENTS = [
    "BLOCK* NameSystem.delete: {blk} is added to invalidSet of {ip}:{port}",
    "PacketResponder {n} for block {blk} Interrupted",
    "Exception in receiveBlock for block {blk} java.io.IOException: Connection reset by peer",
    "BLOCK* NameSystem.addStoredBlock: Redundant addStoredBlock request received for {blk}",
    "WARN dfs.DataNode: writeBlock {blk} received exception java.net.SocketTimeoutException",
]


def rand_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def rand_blk():
    return "blk_" + "".join(random.choices(string.digits, k=10))


def fill(template, blk):
    return template.format(
        blk=blk,
        ip=rand_ip(),
        ip2=rand_ip(),
        port=random.randint(1024, 65000),
        port2=random.randint(1024, 65000),
        n=random.randint(0, 2),
        size=random.randint(1000, 90000000),
    )


def generate_session(start_time, anomaly=False):
    blk = rand_blk()
    n_events = random.randint(4, 6)
    events = random.sample(NORMAL_EVENTS, k=min(n_events, len(NORMAL_EVENTS)))

    if anomaly:
        # inject 1-2 anomalous events at random positions, and sometimes
        # drop a normal event to simulate an incomplete sequence
        for _ in range(random.randint(1, 2)):
            pos = random.randint(0, len(events))
            events.insert(pos, random.choice(ANOMALY_EVENTS))
        if random.random() < 0.5 and len(events) > 3:
            del events[random.randint(0, len(events) - 1)]

    lines = []
    t = start_time
    for ev in events:
        t += timedelta(milliseconds=random.randint(5, 400))
        ts = t.strftime("%y%m%d %H%M%S")
        pid = random.randint(1000, 9999)
        level = random.choice(["INFO", "INFO", "INFO", "WARN"]) if not anomaly else random.choice(["INFO", "WARN", "ERROR"])
        msg = fill(ev, blk)
        lines.append(f"{ts} {pid} {level} dfs.DataNode: {msg}")
    return blk, lines, int(anomaly)


def main(out_path=DATA_DIR / "raw_logs.log",
         label_path=DATA_DIR / "labels.csv",
         n_sessions=2000, anomaly_rate=0.08):
    start = datetime(2026, 1, 1, 0, 0, 0)
    all_lines = []
    labels = [("block_id", "label")]

    for i in range(n_sessions):
        start += timedelta(seconds=random.randint(1, 5))
        is_anom = random.random() < anomaly_rate
        blk, lines, label = generate_session(start, anomaly=is_anom)
        all_lines.extend(lines)
        labels.append((blk, label))

    out_path = Path(out_path)
    label_path = Path(label_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("\n".join(all_lines) + "\n")

    import csv
    with open(label_path, "w", newline="") as f:
        csv.writer(f).writerows(labels)

    n_anom = sum(l for _, l in labels[1:])
    print(f"Wrote {len(all_lines)} log lines across {n_sessions} sessions to {out_path}")
    print(f"Wrote {n_sessions} labels to {label_path} ({n_anom} anomalous, {n_anom/n_sessions:.1%})")


if __name__ == "__main__":
    main()
