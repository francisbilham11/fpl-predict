from __future__ import annotations
import os, json, requests
from ..utils.logging import get_logger
from ..config import settings
from ..auth.login import get_auth_header_candidates

log = get_logger(__name__)
API = "https://fantasy.premierleague.com/api"


def sync_myteam(entry_id: int | None = None) -> None:
    entry = entry_id or settings.FPL_ENTRY_ID
    if not entry:
        raise RuntimeError("No entry id. Pass --entry or set FPL_ENTRY_ID in .env.")

    url = f"{API}/my-team/{int(entry)}/"
    # Try every configured auth method in preference order (token, then cookie) — a
    # stale token shouldn't block a request when a valid cookie is also configured.
    r = None
    for headers in get_auth_header_candidates():
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code not in (401, 403):
            break

    if r.status_code in (401, 403):
        raise RuntimeError("Forbidden (403). Your token/cookie is invalid or expired.")
    r.raise_for_status()

    data = r.json()
    out = "data/processed/myteam_latest.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    log.info("My-team snapshot written → %s", out)