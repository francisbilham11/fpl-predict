from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from ..data.fpl_api import get_bootstrap
from ..utils.cache import DATA
from ..utils.logging import get_logger
log = get_logger(__name__)
def _now() -> str: return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def snapshot_prices() -> Path:
    els = get_bootstrap().get("elements", [])
    df = pd.DataFrame([{"id": e["id"], "web_name": e["web_name"], "now_cost": e["now_cost"], "team": e["team"]} for e in els])
    df["snapshot"] = _now()
    outdir = DATA / "processed" / "prices"; outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"snapshot_{df['snapshot'].iloc[0].replace(':','-')}.csv"; df.to_csv(path, index=False); return path
def show_price_changes() -> None:
    outdir = DATA / "processed" / "prices"; outdir.mkdir(parents=True, exist_ok=True)
    snaps = sorted(outdir.glob("snapshot_*.csv"))
    if not snaps:
        path = snapshot_prices(); log.info("Created first price snapshot: %s", path.name); return
    if len(snaps)==1:
        snapshot_prices(); snaps = sorted(outdir.glob("snapshot_*.csv"))
    prev, curr = snaps[-2], snaps[-1]
    a, b = pd.read_csv(prev), pd.read_csv(curr)
    m = a.merge(b, on=["id","web_name","team"], suffixes=("_prev","_curr"))
    m["delta"] = m["now_cost_curr"] - m["now_cost_prev"]
    changed = m[m["delta"] != 0]
    if changed.empty:
        log.info("No price changes since last snapshot (%s → %s)", prev.name, curr.name)
    else:
        for _, r in changed.iterrows():
            sign = "+" if r["delta"]>0 else ""
            log.info("%s: %s%d (team %s)", r["web_name"], sign, int(r["delta"]), r["team"])
