"""Strict canonical JSON and durable, digest-enveloped publication."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def _check(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, (list, tuple)):
        for x in value:
            _check(x)
        return
    if isinstance(value, dict) and all(type(k) is str for k in value):
        for x in value.values():
            _check(x)
        return
    raise ValueError("non-JSON type, non-string key, or non-finite number")


def encode_canonical(value):
    _check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def loads_strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)
    _check(result)
    return result


def sha256_digest(value):
    return hashlib.sha256(encode_canonical(value)).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_publisher_temp(path):
    return any(p.startswith((".tmp-", ".tmp_")) or
               p.endswith((".tmp", ".new", ".partial", ".lock", ".copy"))
               for p in Path(path).parts)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_record(path):
    path = Path(path)
    try:
        envelope = loads_strict(path.read_bytes())
        if (not isinstance(envelope, dict) or set(envelope) != {"sha256", "data"}
                or envelope["sha256"] != sha256_digest(envelope["data"])):
            raise ValueError("invalid envelope or digest mismatch")
        return envelope["data"]
    except (ValueError, UnicodeError, TypeError) as exc:
        raise ValueError(f"corrupt record {path}: {exc}") from exc


def load_json_or_record(path):
    """Load either a SHA256-enveloped record or a plain JSON file."""
    path = Path(path)
    parsed = loads_strict(path.read_bytes())
    if isinstance(parsed, dict) and set(parsed) == {"sha256", "data"}:
        if parsed["sha256"] != sha256_digest(parsed["data"]):
            raise ValueError(f"digest mismatch in {path}")
        return parsed["data"]
    return parsed


def write_record(path, data, *, overwrite=False):
    path = Path(path)
    raw = encode_canonical({"sha256": sha256_digest(data), "data": data}) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", delete=False) as tmp:
        temp = Path(tmp.name)
        try:
            tmp.write(raw)
            tmp.flush()
            os.fsync(tmp.fileno())
            if overwrite:
                os.replace(temp, path)
            else:
                try:
                    os.link(temp, path)
                except FileExistsError:
                    if encode_canonical(read_record(path)) != encode_canonical(data):
                        raise ValueError(f"write-once conflict: {path}")
            temp.unlink(missing_ok=True)
            fsync_directory(path.parent)
        finally:
            temp.unlink(missing_ok=True)
    return path
