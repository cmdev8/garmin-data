from __future__ import annotations


PAGE_TITLE = "Garmin futóelemzés"

TECH_TERMS = {
    "confidence": "confidence",
    "score": "score",
    "feature": "feature",
    "feature_importance": "feature importance",
    "prediction": "prediction",
    "baseline": "baseline",
    "improvement": "improvement",
    "model": "model",
    "training_load": "training load",
    "overload_risk": "overload risk",
    "diminishing_returns_factor": "diminishing returns factor",
}

RUNNING_TERMS = {
    "easy_run": "laza futás",
    "recovery_run": "regeneráló futás",
    "long_run": "hosszú futás",
    "tempo_run": "tempófutás",
    "threshold": "küszöbedzés",
    "intervals": "résztáv / VO2max",
    "weekly_km": "heti km",
    "elevation_gain": "szintemelkedés",
    "hr_zone": "pulzuszóna",
    "pace": "tempó",
    "training_block": "edzésblokk",
}

PAGES_HU = {
    "overview": "Áttekintés",
    "data_quality": "Adatminőség",
    "diagnostics": "Diagnosztika",
    "activities": "Edzések",
    "training_load": "Training load",
    "fitness": "Fitness / fáradtság",
    "performance_change": "Teljesítménytrend",
    "blocks": "Edzésblokkok",
    "attribution": "ML feature importance",
    "diminishing_returns": "Diminishing returns",
    "predictions_plan": "Prediction és edzésterv",
    "algorithms": "Dokumentáció",
}

WEEKDAYS_HU = {
    "Monday": "Hétfő",
    "Tuesday": "Kedd",
    "Wednesday": "Szerda",
    "Thursday": "Csütörtök",
    "Friday": "Péntek",
    "Saturday": "Szombat",
    "Sunday": "Vasárnap",
}

def hu_day(day: str) -> str:
    return WEEKDAYS_HU.get(day, day)


def plan_family_label(family: object) -> str:
    value = str(family or "n/a")
    return {
        "recovery_week": "regeneráló hét",
        "base_build_week": "alapozó hét",
        "threshold_focus_week": "küszöbfókuszú hét",
        "vo2_focus_week": "VO2max fókuszú hét",
        "speed_support_week": "lendület / futótechnika fókuszú hét",
        "long_run_focus_week": "hosszú futás fókuszú hét",
        "run_walk_progression_week": "futás-séta progresszió",
        "maintenance_week": "szinten tartó hét",
        "fallback_recovery": "biztonsági regeneráló hét",
        "unknown": "ismeretlen",
    }.get(value, value.replace("_", " "))
