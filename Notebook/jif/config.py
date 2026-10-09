"""Paths and credentials shared by all notebooks.

Credentials: environment variables first, then ``Notebook/keys.py``, then defaults.
"""
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # .../Notebook
DATA_DIR = ROOT.parent / "finmcp-analysis" / "data"    # FinMCP-Bench json files
ARTIFACTS = ROOT / "artifacts"
CACHE = ROOT / "cache"
SCHEMA_DIR = ROOT / "schema"

KEY_NAMES = ["TYPESAFE_API_KEY", "JEV_MODEL", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"]
DEFAULTS = {"JEV_MODEL": "jev-1.13.0", "LLM_BASE_URL": "https://api.deepseek.com"}


def load_keys() -> dict:
    """Environment variables, then keys.py, then defaults; also exports the Jev key for the SDK."""
    from_file = {}
    path = ROOT / "keys.py"
    if path.exists():
        spec = importlib.util.spec_from_file_location("jif_keys", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from_file = {k: str(getattr(module, k, "") or "").strip() for k in KEY_NAMES}
    values = {k: os.environ.get(k, "").strip() or from_file.get(k, "") or DEFAULTS.get(k, "") for k in KEY_NAMES}
    if values["TYPESAFE_API_KEY"]:
        os.environ["TYPESAFE_API_KEY"] = values["TYPESAFE_API_KEY"]
    return values


def describe(keys: dict) -> str:
    masked = {k: (v[:3] + "…" + v[-4:] if "KEY" in k and len(v) > 8 else v or "（未设置）") for k, v in keys.items()}
    return "\n".join(f"  {k:<17} {v}" for k, v in masked.items())


for _d in (ARTIFACTS, CACHE, SCHEMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)
