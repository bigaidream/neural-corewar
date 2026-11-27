"""Single-consumer filesystem RPC for optional multi-GPU worker pools."""
from dataclasses import dataclass
from pathlib import Path
import re
import time
from .artifacts import sha256_digest, read_record, write_record, is_publisher_temp, fsync_directory


def _paths(queue_dir, request_id):
    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
        raise ValueError("invalid request id")
    root = Path(queue_dir)
    return root / "requests" / f"{request_id}.json", root / "responses" / f"{request_id}.json"


def _verify(record, request_id, digest=None):
    if record.get("request_id") != request_id or (digest and record.get("request_digest") != digest):
        raise ValueError("request identity/payload conflict")
    return record


def enqueue_request(queue_dir, op, payload, *, request_id=None):
    if not isinstance(op, str) or not isinstance(payload, dict):
        raise ValueError("op must be a string; payload must be an object")
    digest = sha256_digest({"op": op, "payload": payload})
    request_id = request_id or digest
    request, response = _paths(queue_dir, request_id)
    response.parent.mkdir(parents=True, exist_ok=True)
    if response.exists():
        _verify(read_record(response), request_id, digest)
        if request.exists():
            _verify(read_record(request), request_id, digest)
            request.unlink(missing_ok=True)
            fsync_directory(request.parent)
        return request_id
    if request.exists():
        _verify(read_record(request), request_id, digest)
        return request_id
    record = {"request_id": request_id, "request_digest": digest,
              "op": op, "payload": payload, "created_ns": time.time_ns()}
    try:
        write_record(request, record)
    except ValueError:
        if not request.exists():
            raise
        _verify(read_record(request), request_id, digest)
    return request_id


def wait_for_response(queue_dir, request_id, *, timeout=900.0, poll_interval=0.05):
    _, response = _paths(queue_dir, request_id)
    deadline = time.monotonic() + timeout
    while True:
        if response.exists():
            return _verify(read_record(response), request_id)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"RPC response pending: {request_id}")
        time.sleep(poll_interval)


def call_rpc(queue_dir, op, payload, *, request_id=None, timeout=900.0, poll_interval=0.05):
    rid = enqueue_request(queue_dir, op, payload, request_id=request_id)
    return wait_for_response(queue_dir, rid, timeout=timeout, poll_interval=poll_interval)


def rpc_unwrap(queue_dir, op, payload, *, timeout=900.0):
    response = call_rpc(queue_dir, op, payload, timeout=timeout, poll_interval=0.2)
    if response["status"] != "ok":
        raise RuntimeError(f"RPC {op} failed: {response.get('error_type')}")
    return response["result"]


@dataclass(frozen=True)
class PollResult:
    processed: int
    shutdown: bool


def poll_and_serve_once(queue_dir, handlers):
    root = Path(queue_dir)
    (root / "requests").mkdir(parents=True, exist_ok=True)
    (root / "responses").mkdir(parents=True, exist_ok=True)
    batch = []
    for path in (root / "requests").glob("*.json"):
        if is_publisher_temp(path.name):
            continue
        try:
            req = read_record(path)
        except FileNotFoundError:
            continue
        digest = sha256_digest({"op": req["op"], "payload": req["payload"]})
        _verify(req, path.stem, digest)
        batch.append(req)
    batch.sort(key=lambda r: (r["op"] == "shutdown", r["created_ns"], r["request_id"]))
    stopping = False
    for req in batch:
        rid = req["request_id"]
        request, response = _paths(root, rid)
        if response.exists():
            _verify(read_record(response), rid, req["request_digest"])
        else:
            result = {"request_id": rid, "request_digest": req["request_digest"]}
            try:
                value = {"shutdown": True} if req["op"] == "shutdown" else handlers[req["op"]](req["payload"])
                result.update(status="ok", result=value)
            except Exception as exc:
                result.update(status="error", error_type=type(exc).__name__)
            write_record(response, result)
        request.unlink(missing_ok=True)
        fsync_directory(request.parent)
        stopping |= req["op"] == "shutdown"
    return PollResult(len(batch), stopping)


def serve_forever(queue_dir, handlers, *, poll_interval=0.05):
    while True:
        result = poll_and_serve_once(queue_dir, handlers)
        if result.shutdown:
            return
        if not result.processed:
            time.sleep(poll_interval)
