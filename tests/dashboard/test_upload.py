from __future__ import annotations

import inspect

from garmin_runner.dashboard.analysis_runner import run_analysis_for_uploads
from garmin_runner.dashboard.upload import UploadedFilePayload, sanitize_uploaded_filename, validate_uploaded_files


def test_sanitize_uploaded_filename_removes_path():
    name = sanitize_uploaded_filename(1, "../../evil.fit")

    assert name == "0001_evil.fit"
    assert "/" not in name
    assert "\\" not in name


def test_validate_accepts_fit_file():
    errors = validate_uploaded_files([UploadedFilePayload("run.fit", "0001_run.fit", b"abc", 3)])

    assert errors == []


def test_validate_rejects_non_fit_file():
    errors = validate_uploaded_files([UploadedFilePayload("run.txt", "0001_run.fit", b"abc", 3)])

    assert any("csak .fit" in error for error in errors)


def test_validate_rejects_empty_file():
    errors = validate_uploaded_files([UploadedFilePayload("run.fit", "0001_run.fit", b"", 0)])

    assert any("üres" in error for error in errors)


def test_upload_analysis_always_includes_records():
    source = inspect.getsource(run_analysis_for_uploads)

    assert "include_records=True" in source
    assert "detailed_charts=True" in source
    assert "include_records=upload_request.detailed_charts" not in source
