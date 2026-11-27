"""Standard OpenAI-compatible & vLLM HTTP JSON writer transport.

Supports any OpenAI Chat Completions (`/v1/chat/completions`) or Responses
(`/v1/responses`) endpoint—including hosted frontier models (e.g., GPT-6-Astra)
and local vLLM servers (e.g., Inkling-2-Small), as well as heterogeneous A-vs-B
pairings where Player A and Player B use distinct models or endpoints.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .artifacts import encode_canonical, read_record, write_record, sha256_digest


@dataclass(frozen=True)
class WriterConfig:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    wire_api: str = "chat_completions"  # "chat_completions" or "responses"
    reasoning_effort: str | None = "high"
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int = 16384
    timeout_seconds: int = 600
    max_transport_retries: int = 2

    @classmethod
    def from_env(cls, prefix="COREWAR_WRITER"):
        return cls(
            model=os.environ.get(f"{prefix}_MODEL", os.environ.get("OPENAI_MODEL", "GPT-6-Astra")),
            base_url=os.environ.get(f"{prefix}_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")),
            api_key_env=os.environ.get(f"{prefix}_API_KEY_ENV", "OPENAI_API_KEY"),
            wire_api=os.environ.get(f"{prefix}_WIRE_API", "chat_completions"),
            reasoning_effort=os.environ.get(f"{prefix}_REASONING_EFFORT", "high") or None,
            temperature=float(os.environ[f"{prefix}_TEMPERATURE"]) if f"{prefix}_TEMPERATURE" in os.environ else None,
            top_p=float(os.environ[f"{prefix}_TOP_P"]) if f"{prefix}_TOP_P" in os.environ else None,
            max_output_tokens=int(os.environ.get(f"{prefix}_MAX_TOKENS", "16384")),
        )


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


class OpenAIJSONWriter:
    """Receipt-persisting JSON writer compatible with vLLM and OpenAI endpoints."""

    def __init__(self, config: WriterConfig):
        if config.wire_api not in ("chat_completions", "responses"):
            raise ValueError("wire_api must be 'chat_completions' or 'responses'")
        self.config = config

    def _build_request(self, prompt_text: str, instructions: str):
        cfg = self.config
        base = cfg.base_url.rstrip("/")
        api_key = os.environ.get(cfg.api_key_env, "EMPTY")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if cfg.wire_api == "responses":
            url = f"{base}/responses"
            body = {
                "model": cfg.model,
                "max_output_tokens": cfg.max_output_tokens,
                "text": {"format": {"type": "json_object"}},
                "input": [
                    {"role": "developer", "content": instructions},
                    {"role": "user", "content": prompt_text},
                ],
            }
            if cfg.reasoning_effort:
                body["reasoning"] = {"effort": cfg.reasoning_effort}
        else:
            url = f"{base}/chat/completions"
            body = {
                "model": cfg.model,
                "max_tokens": cfg.max_output_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt_text},
                ],
            }
            if cfg.reasoning_effort:
                body["reasoning_effort"] = cfg.reasoning_effort
            if cfg.temperature is not None:
                body["temperature"] = cfg.temperature
            if cfg.top_p is not None:
                body["top_p"] = cfg.top_p
        return url, headers, body

    def _parse_response(self, raw_doc: dict):
        if self.config.wire_api == "responses":
            texts = []
            reasoning = []
            for item in raw_doc.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "output_text":
                            texts.append(part.get("text", ""))
                elif item.get("type") == "reasoning":
                    for part in item.get("summary", []):
                        if isinstance(part, dict) and part.get("text"):
                            reasoning.append(part["text"])
            return {
                "status": raw_doc.get("status", "completed"),
                "text": _strip_fences("".join(texts)),
                "reasoning_text": "\n".join(reasoning) if reasoning else None,
                "usage": raw_doc.get("usage"),
                "response_id": raw_doc.get("id"),
                "reported_model": raw_doc.get("model", self.config.model),
            }
        choices = raw_doc.get("choices") or []
        if not choices:
            return {"status": "empty", "text": "", "reasoning_text": None, "usage": raw_doc.get("usage")}
        msg = choices[0].get("message") or {}
        finish = choices[0].get("finish_reason", "stop")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        return {
            "status": "completed" if finish in ("stop", None) and (msg.get("content") or "").strip() else f"incomplete_{finish}",
            "text": _strip_fences(msg.get("content") or ""),
            "reasoning_text": reasoning if isinstance(reasoning, str) else None,
            "usage": raw_doc.get("usage"),
            "response_id": raw_doc.get("id"),
            "reported_model": raw_doc.get("model", self.config.model),
        }

    def __call__(self, directory, view, instructions):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        prompt_text = encode_canonical(view).decode("ascii")
        binding = {
            "view_sha256": sha256_digest(view),
            "instructions_sha256": sha256_digest(instructions),
            "model": self.config.model,
            "wire_api": self.config.wire_api,
        }
        manifest = directory / "attempts.json"
        if manifest.exists():
            prior = read_record(manifest)
            if prior["binding"] != binding:
                raise ValueError("writer resume binding mismatch")
            return read_record(directory / f"attempt-{prior['final_attempt']}.json")["outcome"]

        url, headers, body = self._build_request(prompt_text, instructions)
        payload_bytes = json.dumps(body).encode("utf-8")
        attempts = []
        outcome = {"status": "transport_error", "text": "", "reasoning_text": None, "usage": None}
        for i in range(self.config.max_transport_retries + 1):
            attempt_file = directory / f"attempt-{i}.json"
            if attempt_file.exists():
                outcome = read_record(attempt_file)["outcome"]
            else:
                try:
                    req = Request(url, data=payload_bytes, headers=headers, method="POST")
                    with urlopen(req, timeout=self.config.timeout_seconds) as resp:
                        raw_doc = json.loads(resp.read().decode("utf-8"))
                    outcome = self._parse_response(raw_doc)
                except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                    outcome = {
                        "status": "transport_error",
                        "error_type": type(exc).__name__,
                        "text": "",
                        "reasoning_text": None,
                        "usage": None,
                    }
                write_record(attempt_file, {"attempt": i, "binding": binding, "outcome": outcome})
            attempts.append({"attempt": i, "status": outcome["status"], "outcome_sha256": sha256_digest(outcome)})
            if outcome.get("status") == "completed" and outcome.get("text", "").strip():
                break
            if i < self.config.max_transport_retries:
                time.sleep(5 * (i + 1))

        write_record(manifest, {"binding": binding, "attempts": attempts, "final_attempt": len(attempts) - 1})
        return outcome
