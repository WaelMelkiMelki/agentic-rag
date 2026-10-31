"""Check API credentials and service connectivity from a .env file.

Run from the repository root with:
    python3 scripts/check_credentials.py
    python3 scripts/check_credentials.py /path/to/.env

Secret values are never printed.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TIMEOUT_SECONDS = 15


def read_env(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without displaying their values."""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def request_status(label: str, url: str, headers: dict[str, str] | None = None) -> bool:
    request = Request(url, headers=headers or {})
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            print(f"PASS  {label} (HTTP {response.status})")
            return True
    except HTTPError as error:
        if error.code in (401, 403):
            detail = "credentials rejected or access denied"
        else:
            detail = f"HTTP {error.code}"
        print(f"FAIL  {label} ({detail})")
    except (URLError, TimeoutError, OSError) as error:
        # Do not print exception text; it can contain request details.
        kind = "timeout" if isinstance(error, TimeoutError) else "connection error"
        print(f"FAIL  {label} ({kind})")
    return False


def check_key(values: dict[str, str], name: str) -> str | None:
    value = values.get(name, "").strip()
    if not value or value in {"your_key_here", "pk-lf-xxx", "sk-lf-xxx"}:
        print(f"FAIL  {name} (missing or still a placeholder)")
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_env = Path(__file__).resolve().parents[1] / ".env"
    parser.add_argument("env_file", nargs="?", type=Path, default=default_env)
    args = parser.parse_args()

    if not args.env_file.is_file():
        print(f"FAIL  .env file not found: {args.env_file}")
        return 1

    try:
        values = read_env(args.env_file)
    except OSError:
        print("FAIL  could not read .env file")
        return 1

    results: list[bool] = []

    gemini_key = check_key(values, "GEMINI_API_KEY")
    if gemini_key:
        results.append(request_status(
            "Gemini API key",
            "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1",
            {"x-goog-api-key": gemini_key},
        ))

    openrouter_key = check_key(values, "OPENROUTER_API_KEY")
    if openrouter_key:
        results.append(request_status(
            "OpenRouter API key",
            "https://openrouter.ai/api/v1/key",
            {"Authorization": f"Bearer {openrouter_key}"},
        ))

    public_key = check_key(values, "LANGFUSE_PUBLIC_KEY")
    secret_key = check_key(values, "LANGFUSE_SECRET_KEY")
    if public_key and secret_key:
        host = values.get("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        results.append(request_status(
            "Langfuse API keys",
            f"{host}/api/public/projects",
            {"Authorization": f"Basic {token}"},
        ))

    qdrant_url = values.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
    results.append(request_status("Qdrant service", f"{qdrant_url}/readyz"))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} configured service checks passed.")
    return 0 if total and passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
