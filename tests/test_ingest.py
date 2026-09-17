"""Match ingestion.

The regression these guard against: at the 2026/27 rollover, `ingest_full` rebuilt the
combined match file from only the seasons fetched in that run. The new season had no finished
fixtures, so the FPL-API branch returned nothing, and the combined file was rewritten without
the 2025/26 season that was already on disk.
"""

import pandas as pd

from src.fpl_predict.data import ingest
from src.fpl_predict.data.ingest import (
    MATCH_COLS,
    _combine_stored,
    _current_season_matches,
    _finalise,
)
from src.fpl_predict.utils.io import write_parquet


def _matches(season: int, n: int = 3) -> pd.DataFrame:
    return _finalise(
        pd.DataFrame(
            {
                "date": pd.date_range(f"{season}-08-15", periods=n, freq="7D", tz="UTC"),
                "gameweek": range(1, n + 1),
                "home_team": ["Man City", "Arsenal", "Liverpool"][:n],
                "away_team": ["Chelsea", "Spurs", "Everton"][:n],
                "home_goals": [2, 1, 3][:n],
                "away_goals": [0, 1, 1][:n],
                "match_id": range(1, n + 1),
                "status": "FINISHED",
            }
        ),
        season,
    )


def test_finalise_produces_the_canonical_schema():
    df = _matches(2025)
    assert list(df.columns) == MATCH_COLS
    assert df["home_slug"].tolist() == ["man_city", "arsenal", "liverpool"]
    assert df["away_slug"].tolist() == ["chelsea", "spurs", "everton"]


def test_finalise_maps_football_data_spellings_to_the_same_slugs():
    fd = _finalise(
        pd.DataFrame(
            {
                "date": ["2025-08-15T19:00:00Z"],
                "home_team": ["Manchester City FC"],
                "away_team": ["Tottenham Hotspur FC"],
                "home_goals": [2],
                "away_goals": [1],
                "match_id": [1],
                "status": ["FINISHED"],
            }
        ),
        2025,
    )
    assert fd["home_slug"].iloc[0] == "man_city"
    assert fd["away_slug"].iloc[0] == "spurs"


def test_finalise_drops_rows_without_a_resolvable_date_or_team():
    df = _finalise(
        pd.DataFrame(
            {
                "date": ["2025-08-15T19:00:00Z", None],
                "home_team": ["Man City", "Arsenal"],
                "away_team": ["Chelsea", "Spurs"],
                "home_goals": [2, 1],
                "away_goals": [0, 1],
                "match_id": [1, 2],
                "status": ["FINISHED", "FINISHED"],
            }
        ),
        2025,
    )
    assert len(df) == 1


def test_finalise_on_empty_input_returns_the_schema_not_an_error():
    df = _finalise(pd.DataFrame(), 2026)
    assert list(df.columns) == MATCH_COLS
    assert df.empty


def test_combine_reads_every_stored_season(tmp_path):
    for season in (2023, 2024, 2025):
        write_parquet(_matches(season), tmp_path / f"EPL_{season}_matches.parquet")

    combined = _combine_stored(tmp_path, "EPL")
    assert sorted(combined["season"].unique()) == [2023, 2024, 2025]
    assert len(combined) == 9


def test_combine_ignores_its_own_output_file(tmp_path):
    write_parquet(_matches(2025), tmp_path / "EPL_2025_matches.parquet")
    write_parquet(_matches(2025), tmp_path / "EPL_all_matches.parquet")

    combined = _combine_stored(tmp_path, "EPL")
    assert len(combined) == 3, "the combined file must not be folded back into itself"


def test_combine_deduplicates_the_same_match_twice(tmp_path):
    write_parquet(_matches(2025), tmp_path / "EPL_2025_matches.parquet")
    # Same fixtures arriving from a second source with different ids.
    dup = _matches(2025)
    dup["match_id"] = [101, 102, 103]
    write_parquet(dup, tmp_path / "EPL_2025b_matches.parquet")

    assert len(_combine_stored(tmp_path, "EPL")) == 3


def test_combine_skips_files_predating_the_slug_schema(tmp_path):
    write_parquet(_matches(2025), tmp_path / "EPL_2025_matches.parquet")
    legacy = pd.DataFrame(
        {
            "season": [2024],
            "date": pd.to_datetime(["2024-08-16T19:00:00Z"]),
            "home_team": ["Man City"],
            "away_team": ["Chelsea"],
            "home_goals": [2],
            "away_goals": [0],
            "match_id": [1],
        }
    )
    write_parquet(legacy, tmp_path / "EPL_2024_matches.parquet")

    combined = _combine_stored(tmp_path, "EPL")
    assert sorted(combined["season"].unique()) == [2025]


def test_empty_current_season_fetch_does_not_erase_stored_history(tmp_path, monkeypatch):
    """The exact August failure: a new season with no finished fixtures yet."""
    outdir = tmp_path / "football-data"
    outdir.mkdir(parents=True)
    for season in (2024, 2025):
        write_parquet(_matches(season), outdir / f"EPL_{season}_matches.parquet")

    monkeypatch.setattr(ingest, "RAW", tmp_path)
    monkeypatch.setattr(ingest, "_completed_season_matches", lambda season, lg: pd.DataFrame())
    monkeypatch.setattr(ingest, "_current_season_matches", lambda season: pd.DataFrame())

    combined = ingest.ingest_full(seasons=[2024, 2025])

    assert sorted(combined["season"].unique()) == [2024, 2025]
    assert (outdir / "EPL_2024_matches.parquet").exists()
    assert (outdir / "EPL_2025_matches.parquet").exists()


def test_current_season_fetch_ignores_unfinished_and_scoreless_fixtures(monkeypatch):
    bootstrap = {
        "teams": [
            {"id": 1, "name": "Arsenal", "short_name": "ARS"},
            {"id": 2, "name": "Chelsea", "short_name": "CHE"},
        ]
    }
    fixtures = [
        # finished with a score: kept
        {
            "finished": True,
            "kickoff_time": "2026-08-15T14:00:00Z",
            "event": 1,
            "team_h": 1,
            "team_a": 2,
            "team_h_score": 2,
            "team_a_score": 1,
            "id": 1,
        },
        # not finished: dropped
        {
            "finished": False,
            "kickoff_time": "2026-08-22T14:00:00Z",
            "event": 2,
            "team_h": 2,
            "team_a": 1,
            "team_h_score": None,
            "team_a_score": None,
            "id": 2,
        },
        # flagged finished but no score yet: dropped rather than read as 0-0
        {
            "finished": True,
            "kickoff_time": "2026-08-29T14:00:00Z",
            "event": 3,
            "team_h": 1,
            "team_a": 2,
            "team_h_score": None,
            "team_a_score": None,
            "id": 3,
        },
    ]
    monkeypatch.setattr(ingest, "get_bootstrap", lambda: bootstrap)
    monkeypatch.setattr(ingest, "get_fixtures", lambda: fixtures)

    df = _current_season_matches(2026)
    assert len(df) == 1
    assert df["home_slug"].iloc[0] == "arsenal"
    assert df["home_goals"].iloc[0] == 2


def test_current_season_fetch_survives_an_api_failure(monkeypatch):
    def boom():
        raise RuntimeError("API down")

    monkeypatch.setattr(ingest, "get_fixtures", boom)
    assert _current_season_matches(2026).empty
