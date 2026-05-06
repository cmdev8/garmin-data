from __future__ import annotations

from html import escape
from typing import Mapping


def show_warnings(warnings: list[str]) -> None:
    import streamlit as st

    for warning in warnings:
        st.warning(warning)


def metric_grid(values: Mapping[str, object], columns: int = 4) -> None:
    import streamlit as st

    cols = st.columns(columns)
    for index, (label, value) in enumerate(values.items()):
        cols[index % columns].markdown(_metric_card_html(str(label), str(value)), unsafe_allow_html=True)


def _metric_card_html(label: str, value: str) -> str:
    return f"""
<div style="min-height: 5.4rem; padding: 0.1rem 0;">
  <div style="font-size: 0.92rem; color: rgba(49, 51, 63, 0.82); line-height: 1.25; margin-bottom: 0.25rem;">
    {escape(label)}
  </div>
  <div style="font-size: clamp(1.35rem, 2.4vw, 2rem); line-height: 1.12; font-weight: 400; overflow-wrap: anywhere; word-break: normal;">
    {escape(value)}
  </div>
</div>
"""
