"""Download AMFI monthly category reports into data/raw/.

Usage (from project root):
    python src\\download_amfi.py              # normal, verified HTTPS
    python src\\download_amfi.py --insecure   # only if you get CERTIFICATE_VERIFY_FAILED

URL pattern: https://portal.amfiindia.com/spages/am<mon><year>repo.xls
"""
import argparse
import os
import time
from datetime import date

import requests

MON = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
BASE = "https://portal.amfiindia.com/spages/am{mon}{year}repo.xls"
XLS_MAGIC = b"\xd0\xcf\x11\xe0"
XLSX_MAGIC = b"PK\x03\x04"


def month_range(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def url_for(y, m):
    return BASE.format(mon=MON[m - 1], year=y)


def is_excel(data):
    return data[:4] in (XLS_MAGIC, XLSX_MAGIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-04")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--insecure", action="store_true")
    a = ap.parse_args()
    if a.insecure:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        print("NOTE: SSL verification is OFF (public AMFI files only).")
    s = tuple(int(x) for x in a.start.split("-"))
    e = tuple(int(x) for x in a.end.split("-"))
    os.makedirs(a.out, exist_ok=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0"
    ok, failed, log = 0, [], []
    for y, m in month_range(s, e):
        url = url_for(y, m)
        path = os.path.join(a.out, f"amfi_{y}_{m:02d}.xls")
        if os.path.exists(path):
            with open(path, "rb") as f:
                if is_excel(f.read(4)):
                    ok += 1
                    continue
        try:
            r = sess.get(url, timeout=60, verify=not a.insecure)
            r.raise_for_status()
            if not is_excel(r.content):
                raise ValueError("response is not an Excel file")
            with open(path, "wb") as f:
                f.write(r.content)
            ok += 1
            log.append(f"| AMFI monthly report {y}-{m:02d} | {url} | {date.today().isoformat()} | AMFI public data; cite AMFI |")
            print("downloaded", y, m)
        except requests.exceptions.SSLError:
            print("SSL ERROR - rerun with:  python src\\download_amfi.py --insecure")
            return
        except Exception as ex:
            failed.append((y, m, url, str(ex)))
            print("FAILED    ", y, m, ex)
        time.sleep(0.7)
    if log:
        with open(os.path.join(a.out, "SOURCES.md"), "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(log) + "\n")
    print(f"\nDone. {ok} files present, {len(failed)} failed.")
    for y, m, url, ex in failed:
        print(f"  {y}-{m:02d}  {url}  ({ex})")


if __name__ == "__main__":
    main()
