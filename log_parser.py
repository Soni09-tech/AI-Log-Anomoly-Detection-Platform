"""
Log parsing: text -> (block_id, event_template_id, timestamp)

This is a lightweight, dependency-free re-implementation of the core idea
behind Drain (the log-parsing algorithm used in most anomaly-detection
papers, incl. Loghub / DeepLog): mask out variable tokens (numbers, IPs,
hex, block ids) so that lines that only differ in their variable parts
collapse onto the same "event template".

HDFS-style lines retain their block/session IDs. Other text lines are accepted
as generic events and assigned to one file-level session when no block ID is
present. For production log parsing, this can be replaced with `drain3`:
    pip install drain3
    from drain3 import TemplateMiner
    miner = TemplateMiner()
    result = miner.add_log_message(line)
    template_id = result["cluster_id"]
Drain3 is more robust (persistent state, better clustering, streaming
support) but the offline logic here is enough to build and validate the
full pipeline before you have internet access to install it.
"""

import csv
import codecs
import re
from collections import OrderedDict
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" if (ROOT / "data").exists() else ROOT

BLOCK_RE = re.compile(r"blk_-?\d+")
IP_PORT_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?")
NUM_RE = re.compile(r"\b\d+\b")
TIMESTAMP_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})[T ]"
    r"(?P<time>\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)
LEVEL_RE = re.compile(r"\b(TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\b", re.IGNORECASE)
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


def decode_log_bytes(raw_bytes: bytes) -> str:
    """Decode common text uploads and reject clearly binary content."""
    if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw_bytes.decode("utf-16", errors="replace")
    else:
        text = raw_bytes.decode("utf-8-sig", errors="replace")

    controls = sum(
        ord(character) < 32 and character not in "\t\n\r\f"
        for character in text
    )
    if "\x00" in text or controls > len(text) * 0.01:
        raise ValueError(
            "This file appears to be binary. Upload a text-based log, CSV, JSON, or similar file."
        )
    return text


def parse_log_line(
    line: str,
    store: "TemplateStore | None" = None,
) -> dict[str, str | int] | None:
    """Parse one non-empty line into a session event and normalized template."""
    line = line.strip()
    if not line:
        return None
    match = LOG_LINE_RE.match(line)
    if match:
        date = match.group("date")
        time = match.group("time")
        level = match.group("level").upper()
        content = match.group("content")
        block_id = extract_block_id(content)
    else:
        timestamp = TIMESTAMP_RE.search(line)
        date = timestamp.group("date") if timestamp else ""
        time = timestamp.group("time") if timestamp else ""
        level_match = LEVEL_RE.search(line)
        level = level_match.group(1).upper() if level_match else "INFO"
        if level == "WARNING":
            level = "WARN"
        elif level in {"FATAL", "CRITICAL"}:
            level = "ERROR"
        content = line
        block_id = extract_block_id(line)
    template = templatize(content)
    return {
        "block_id": block_id or "file_session_0",
        "event_id": store.get_id(template) if store is not None else -1,
        "level": level,
        "date": date,
        "time": time,
        "content": content,
        "template": template,
    }


def iter_log_chunks(
    stream: BinaryIO,
    lines_per_chunk: int = 25_000,
    read_size: int = 64 * 1024,
    max_line_characters: int = 1_048_576,
    on_bytes_read: Callable[[int], None] | None = None,
) -> Iterator[list[str]]:
    """Yield decoded text lines in bounded batches from a binary stream."""
    if lines_per_chunk < 1 or read_size < 1:
        raise ValueError("Chunk sizes must be positive.")
    prefix = stream.read(4)
    if prefix and on_bytes_read is not None:
        on_bytes_read(len(prefix))
    if prefix.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encoding = "utf-16"
    else:
        encoding = "utf-8-sig"
    decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
    pending = ""
    batch: list[str] = []
    probe = prefix

    while True:
        chunk = probe if probe else stream.read(read_size)
        probe = b""
        if not chunk:
            pending += decoder.decode(b"", final=True)
            break
        if on_bytes_read is not None:
            on_bytes_read(len(chunk))
        decoded = decoder.decode(chunk)
        if "\x00" in decoded:
            raise ValueError("This file appears to be binary; upload a text-based log.")
        if decoded:
            controls = sum(
                ord(character) < 32 and character not in "\t\n\r\f"
                for character in decoded
            )
            if controls > len(decoded) * 0.01:
                raise ValueError("This file appears to be binary; upload a text-based log.")
        pending += decoded
        if len(pending) > max_line_characters and "\n" not in pending:
            raise ValueError(
                f"A log line exceeds the {max_line_characters:,}-character processing limit."
            )
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            if len(line) > max_line_characters:
                raise ValueError(
                    f"A log line exceeds the {max_line_characters:,}-character processing limit."
                )
            batch.append(line.rstrip("\r"))
            if len(batch) >= lines_per_chunk:
                yield batch
                batch = []

    if pending:
        if len(pending) > max_line_characters:
            raise ValueError(
                f"A log line exceeds the {max_line_characters:,}-character processing limit."
            )
        batch.append(pending.rstrip("\r"))
    if batch:
        yield batch


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
    Parses log lines into structured rows:
        block_id, event_id, level, date, time
    Writes them to out_path as CSV, and the template dictionary to
    template_out_path.
    """
    store = TemplateStore()
    row_count = 0

    with open(in_path, "rb") as source, open(out_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["block_id", "event_id", "level", "date", "time", "content"],
        )
        writer.writeheader()
        for chunk in iter_log_chunks(source):
            for line in chunk:
                parsed = parse_log_line(line, store)
                if parsed is None:
                    continue
                writer.writerow({key: value for key, value in parsed.items() if key != "template"})
                row_count += 1

    store.save(template_out_path)
    print(f"Parsed {row_count} lines into {len(store.template_to_id)} event templates")
    print(f"-> {out_path}")
    print(f"-> {template_out_path}")
    return row_count, store


if __name__ == "__main__":
    parse_log_file(
        in_path=DATA_DIR / "raw_logs.log",
        out_path=DATA_DIR / "parsed_logs.csv",
        template_out_path=DATA_DIR / "templates.csv",
    )
