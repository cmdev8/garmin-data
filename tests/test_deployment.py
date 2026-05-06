from __future__ import annotations

from pathlib import Path


def test_streamlit_deployment_uses_supported_python_runtime():
    assert Path("runtime.txt").read_text(encoding="utf-8").strip() == "python-3.11"


def test_streamlit_deployment_pins_modern_reportlab():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "reportlab>=4.2,<5" in requirements
    assert "reportlab>=4.2,<5" in pyproject
    assert "reportlab==3.5.59" not in requirements
    assert "reportlab==3.5.59" not in pyproject
