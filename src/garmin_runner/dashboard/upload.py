from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from garmin_runner.config import WEEKDAYS
from garmin_runner.dashboard.i18n import hu_day

MAX_FILES = 1000
MAX_TOTAL_UPLOAD_MB = 1000
MAX_SINGLE_FILE_MB = 50


@dataclass
class UploadedFilePayload:
    original_name: str
    safe_name: str
    content: bytes
    size_bytes: int


@dataclass
class UploadRequest:
    fit_files: list[UploadedFilePayload]
    athlete_settings: dict
    run_walk_enabled: bool
    detailed_charts: bool = True
    warnings: list[str] = field(default_factory=list)


def sanitize_uploaded_filename(index: int, original_name: str) -> str:
    stem = Path(original_name).stem
    stem = re.sub(r"[\x00-\x1f\x7f]+", "", stem)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = stem.strip("._")[:80] or "activity"
    return f"{index:04d}_{stem}.fit"


def validate_uploaded_files(files: list[UploadedFilePayload]) -> list[str]:
    errors: list[str] = []
    if not files:
        errors.append("Tölts fel legalább egy .fit fájlt.")
    if len(files) > MAX_FILES:
        errors.append(f"Legfeljebb {MAX_FILES} fájlt tölthetsz fel.")
    total = sum(file.size_bytes for file in files)
    if total > MAX_TOTAL_UPLOAD_MB * 1024 * 1024:
        errors.append(f"A teljes feltöltési méret legfeljebb {MAX_TOTAL_UPLOAD_MB} MB lehet.")
    for file in files:
        if not file.original_name.lower().endswith(".fit"):
            errors.append(f"{file.original_name}: csak .fit fájl támogatott.")
        if file.size_bytes <= 0:
            errors.append(f"{file.original_name}: a fájl üres.")
        if file.size_bytes > MAX_SINGLE_FILE_MB * 1024 * 1024:
            errors.append(f"{file.original_name}: a fájl nagyobb mint {MAX_SINGLE_FILE_MB} MB.")
    return errors


def uploaded_files_to_payloads(uploaded_files) -> list[UploadedFilePayload]:
    payloads = []
    for index, uploaded in enumerate(uploaded_files or [], 1):
        content = uploaded.getvalue()
        payloads.append(
            UploadedFilePayload(
                original_name=uploaded.name,
                safe_name=sanitize_uploaded_filename(index, uploaded.name),
                content=content,
                size_bytes=len(content),
            )
        )
    return payloads


def render_upload_panel() -> UploadRequest | None:
    import streamlit as st

    uploaded = st.file_uploader(
        "Garmin FIT fájlok feltöltése",
        type=["fit"],
        accept_multiple_files=True,
        help="Garminból exportált aktivitásokat tölts fel. A nem futás jellegű edzéseket automatikusan kiszűrjük.",
    )
    payloads = uploaded_files_to_payloads(uploaded)
    errors = validate_uploaded_files(payloads)

    st.caption(
        "A feltöltött FIT fájlokat memóriából elemezzük, az app nem tárolja őket. "
        "A feldolgozott összefoglalók letölthetők, de a nyers FIT fájlok nem kerülnek bele."
    )

    with st.form("athlete_settings"):
        target_distance = st.selectbox("Célverseny távja", [1.0, 5.0, 10.0, 21.1, 42.2], index=2)
        day_labels = {hu_day(day): day for day in WEEKDAYS}
        available_day_labels = st.multiselect(
            "Edzésre elérhető napok",
            list(day_labels.keys()),
            default=[hu_day(day) for day in ["Tuesday", "Thursday", "Saturday", "Sunday"]],
        )
        available_days = [day_labels[label] for label in available_day_labels]
        preferred_label = st.selectbox("Hosszú futás preferált napja", list(day_labels.keys()), index=6)
        preferred_long_run_day = day_labels[preferred_label]
        athlete_timezone = st.text_input("Időzóna", value="Europe/Budapest")
        max_hr = st.number_input("Maximális pulzus (opcionális)", min_value=0, max_value=240, value=0)
        resting_hr = st.number_input("Nyugalmi pulzus (opcionális)", min_value=0, max_value=120, value=0)
        lactate_hr = st.number_input("Laktátküszöb pulzus (opcionális)", min_value=0, max_value=240, value=0)
        lactate_pace = st.number_input("Laktátküszöb tempó, mp/km (opcionális)", min_value=0, max_value=1000, value=0)
        max_minutes = st.number_input("Maximális perc hetente (opcionális)", min_value=0, max_value=2000, value=300)
        max_volume_increase = st.number_input("Maximális heti km-emelés %", min_value=0, max_value=50, value=10)
        max_hard_days = st.number_input("Maximális minőségi nap hetente", min_value=0, max_value=4, value=2)
        horizon_days = st.slider("Tervezési horizont napokban", min_value=1, max_value=28, value=7)
        run_walk_enabled = st.toggle("Futás-séta támogatás", value=True)
        include_walk_run_candidates = st.toggle("Séta-futás jelöltek engedélyezése", value=False)
        submitted = st.form_submit_button("Feltöltött fájlok elemzése", disabled=bool(errors))

    if errors:
        for error in errors:
            st.error(error)
        return None
    if not submitted:
        return None

    settings = {
        "athlete": {"timezone": athlete_timezone},
        "goals": {"target_distance_km": target_distance},
        "availability": {
            "available_days": available_days,
            "preferred_long_run_day": preferred_long_run_day,
            "max_minutes_per_week": max_minutes or None,
        },
        "physiology": {
            "max_hr": max_hr or None,
            "resting_hr": resting_hr or None,
            "lactate_threshold_hr": lactate_hr or None,
            "lactate_threshold_pace_s_per_km": lactate_pace or None,
        },
        "training": {"max_hard_days_per_week": max_hard_days, "max_volume_increase_pct": max_volume_increase},
        "run_walk": {
            "enabled": run_walk_enabled,
            "allow_walk_run_candidates": include_walk_run_candidates,
        },
        "planning": {"horizon_days": horizon_days},
    }
    return UploadRequest(
        fit_files=payloads,
        athlete_settings=settings,
        run_walk_enabled=run_walk_enabled,
        detailed_charts=True,
    )
