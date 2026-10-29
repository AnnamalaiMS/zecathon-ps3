"""Build data/processed/flows.csv from the AMFI monthly reports in data/raw/.

Usage (project root):  python src/data_loaders.py

Columns are located by header text (layouts differ between 2019 and 2023+), the
month is taken from the file name AND cross-checked against the report title, and
the loader stops with a clear error on any mismatch instead of guessing."""
import glob
import os
import re
import sys
import warnings

import pandas as pd

TARGETS = {"liquid fund": "liquid", "overnight fund": "overnight"}
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]


def _norm(x):
    return re.sub(r"\s+", " ", str(x)).strip().lower()


def _title_month(raw):
    text = _norm(" ".join(str(v) for v in raw.iloc[:6].values.ravel() if str(v) != "nan"))
    m = re.search(r"\b(" + "|".join(MONTHS) + r")[\s\-,]+(\d{4})\b", text)
    if not m:
        raise ValueError("could not find the report month in the title rows")
    return int(m.group(2)), MONTHS.index(m.group(1)) + 1


def _find_col(header, pattern):
    hits = [i for i, h in enumerate(header) if re.search(pattern, h)]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one column matching {pattern!r}, found {hits}")
    return hits[0]


def parse_report_frame(raw, year, month):
    """raw: sheet read with header=None. Returns one row per target category."""
    ty, tm = _title_month(raw)
    if (ty, tm) != (year, month):
        raise ValueError(f"file says {year}-{month:02d} but report title says {ty}-{tm:02d}")
    hdr_idx = next((i for i in range(min(10, len(raw))) if any("net inflow" in _norm(c) for c in raw.iloc[i])), None)
    if hdr_idx is None:
        raise ValueError("header row with 'Net Inflow' not found")
    header = [_norm(c) for c in raw.iloc[hdr_idx]]
    c_name = _find_col(header, r"^scheme (category|name)")
    c_mob = _find_col(header, r"(mobilized|mobilised)")
    c_red = _find_col(header, r"(repurchase|redemption)")
    c_net = _find_col(header, r"^net inflow")
    c_aum = _find_col(header, r"^(net assets under management|aum) as on")
    out = []
    for key, cat in TARGETS.items():
        rows = raw.iloc[hdr_idx + 1:]
        rows = rows[rows.iloc[:, c_name].map(_norm) == key]
        if len(rows) != 1:
            raise ValueError(f"expected exactly one '{key}' row, found {len(rows)}")
        r = rows.iloc[0]
        vals = {k: pd.to_numeric(r.iloc[c], errors="coerce")
                for k, c in [("mobilized_cr", c_mob), ("redemption_cr", c_red), ("net_flow_cr", c_net), ("aum_cr", c_aum)]}
        if any(pd.isna(v) for v in vals.values()):
            raise ValueError(f"non-numeric value in '{key}' row: {vals}")
        out.append({"month": pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0), "category": cat, **vals})
    return out


def parse_workbook(sheets, year, month):
    """sheets: dict of sheet name -> frame (header=None). Uses the first sheet that parses fully."""
    errs = []
    for name, raw in sheets.items():
        try:
            return parse_report_frame(raw, year, month)
        except Exception as e:
            errs.append(f"[{name}] {e}")
    raise ValueError(" | ".join(errs))


def main(raw_dir="data/raw", out_path="data/processed/flows.csv"):
    files = sorted(glob.glob(os.path.join(raw_dir, "amfi_*.xls")))
    if not files:
        sys.exit(f"No amfi_*.xls files in {raw_dir}")
    rows = []
    for f in files:
        y, m = map(int, re.search(r"amfi_(\d{4})_(\d{2})\.xls$", f).groups())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sheets = pd.read_excel(f, sheet_name=None, header=None)
        try:
            rows += parse_workbook(sheets, y, m)
        except Exception as e:
            sys.exit(f"STOP: {os.path.basename(f)}: {e}")
    df = pd.DataFrame(rows).sort_values(["category", "month"]).reset_index(drop=True)
    for cat, g in df.groupby("category"):
        full = pd.date_range(g.month.min(), g.month.max(), freq="ME")
        missing = full.difference(pd.DatetimeIndex(g.month))
        if len(missing):
            sys.exit(f"STOP: {cat} is missing months: {[d.strftime('%Y-%m') for d in missing]}")
        if (g.aum_cr <= 0).any():
            sys.exit(f"STOP: {cat} has non-positive AUM")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(df)} rows")
    for cat, g in df.groupby("category"):
        print(f"  {cat}: {len(g)} months, {g.month.min():%Y-%m} to {g.month.max():%Y-%m}, "
              f"net flow min {g.net_flow_cr.min():,.0f} / max {g.net_flow_cr.max():,.0f} Rs crore")


if __name__ == "__main__":
    main()