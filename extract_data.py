from __future__ import annotations

from typing import Any

import pandas as pd
import requests


BASE_URL = (
    "https://interactive.guim.co.uk/atoms/labs/2025/09/"
    "university-guide/overview/v/1771858244880/assets/data"
)
OVERVIEW_URL = f"{BASE_URL}/overview.json"


def load_json(url: str, timeout: int = 30) -> dict[str, Any]:
    """Download and return JSON data from a URL."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_overview(timeout: int = 30) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load the overview JSON and main university rankings."""
    data = load_json(OVERVIEW_URL, timeout)
    main_rank = pd.DataFrame(data.get("institutions", []))
    return data, main_rank


def load_subjects(
    subjects: list[dict[str, Any]],
    timeout: int = 30,
) -> dict[str, pd.DataFrame]:
    """Load rankings for each subject."""
    subject_dataframes: dict[str, pd.DataFrame] = {}

    for subject in subjects:
        subject_id = subject["id"]
        subject_name = subject["title"]
        subject_url = f"{BASE_URL}/{subject_id}.json"

        subject_json = load_json(subject_url, timeout)
        records = subject_json.get("institutions", [])

        subject_dataframes[subject_id] = pd.json_normalize(records).assign(
            subject_id=subject_id,
            subject=subject_name,
        )

    return subject_dataframes


def load_data(timeout: int = 30) -> dict[str, Any]:
    """Load overview, main rankings, subjects, and subject rankings."""
    data, main_rank = load_overview(timeout)

    subjects_data = [
        {
            "id": subject["id"],
            "title": subject["title"],
        }
        for subject in data.get("subjects", [])
        if subject.get("id") and subject.get("title")
    ]

    subject_dataframes = load_subjects(subjects_data, timeout)

    return {
        "data": data,
        "main_rank": main_rank,
        "subjects": subjects_data,
        "subject_dataframes": subject_dataframes,
    }