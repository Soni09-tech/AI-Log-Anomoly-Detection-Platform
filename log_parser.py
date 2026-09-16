"""
Log parsing: raw text -> (block_id, event_template_id, timestamp)

This is a lightweight, dependency-free re-implementation of the core idea
behind Drain (the log-parsing algorithm used in most anomaly-detection
papers, incl. Loghub / DeepLog): mask out variable tokens (numbers, IPs,
hex, block ids) so that lines that only differ in their variable parts
collapse onto the same "event template".

For the real project, swap this for the actual `drain3` package:
    pip install drain3
    from drain3 import TemplateMiner
    miner = TemplateMiner()
    result = miner.add_log_message(line)
    template_id = result["cluster_id"]
Drain3 is more robust (persistent state, better clustering, streaming
support) but the offline logic here is enough to build and validate the
full pipeline before you have internet access to install it.
"""

import re
import csv
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" if (ROOT / "data").exists() else ROOT

BLOCK_RE = re.compile(r"blk_-?\d+")
IP_PORT_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?")
NUM_RE = re.compile(r"\b\d+\b")
LOG_LINE_RE = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+(?P<level>\w+)\s+(?P<component>[\w.]+):\s*(?P<content>.*)$"
)


def extract_block_id(content: str):
    m = BLOCK_RE.search(content)
    return m.group(0) if m else None


def templatize(content: str) -> str:
    """Mask variable tokens so structurally-identical lines share a template."""
    t = BLOCK_RE.sub("<BLK>", content)
    t = IP_PORT_RE.sub("<IP>", t)
    t = NUM_RE.sub("<NUM>", t)
    return t


class TemplateStore:
    """Assigns a stable integer id to each distinct template string."""

    def __init__(self):
        self.template_to_id = OrderedDict()

    def get_id(self, template: str) -> int:
        if template not in self.template_to_id:
            self.template_to_id[template] = len(self.template_to_id)
        return self.template_to_id[template]

    def save(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event_id", "template"])
            for tmpl, idx in self.template_to_id.items():
                w.writerow([idx, tmpl])


def parse_log_file(in_path, out_path, template_out_path):
    """
    Parses raw log lines into structured rows:
        block_id, event_id, level, date, time
    Writes them to out_path as CSV, and the template dictionary to
    template_out_path.
    """
    store = TemplateStore()
    rows = []
    skipped = 0

    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = LOG_LINE_RE.match(line)
            if not m:
                skipped += 1
                continue
            content = m.group("content")
            blk = extract_block_id(content)
            if blk is None:
                skipped += 1
                continue
            template = templatize(content)
            event_id = store.get_id(template)
            rows.append({
                "block_id": blk,
                "event_id": event_id,
                "level": m.group("level"),
                "date": m.group("date"),
                "time": m.group("time"),
            })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["block_id", "event_id", "level", "date", "time"])
        w.writeheader()
        w.writerows(rows)

    store.save(template_out_path)
    print(f"Parsed {len(rows)} lines into {len(store.template_to_id)} event templates "
          f"({skipped} unparseable lines skipped)")
    print(f"-> {out_path}")
    print(f"-> {template_out_path}")
    return rows, store


if __name__ == "__main__":
    parse_log_file(
        in_path=DATA_DIR / "raw_logs.log",
        out_path=DATA_DIR / "parsed_logs.csv",
        template_out_path=DATA_DIR / "templates.csv",
    )
