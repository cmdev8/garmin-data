from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def write_json(path: str | Path, data) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def write_markdown(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def write_table(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    try:
        import pandas as pd

        pd.DataFrame(materialized).to_parquet(target, index=False)
    except Exception:
        target.write_text(json.dumps(materialized, indent=2, default=str), encoding="utf-8")


def read_table(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.exists():
        return []
    try:
        import pandas as pd

        return pd.read_parquet(source).to_dict(orient="records")
    except Exception:
        try:
            return json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return []


def read_json(path: str | Path):
    source = Path(path)
    if not source.exists():
        return None
    return json.loads(source.read_text(encoding="utf-8"))
