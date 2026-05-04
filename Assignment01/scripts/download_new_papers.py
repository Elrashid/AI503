"""
Download the 18 new papers (RP08, RP11, RP14, ..., RP50) from arXiv.

Output: v3/papers_pdf/RP{nn}_{Author}_{Year}.pdf  (flat — same dir as the 32 kept PDFs)

Idempotent — skips already-downloaded files.
"""

import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "papers_pdf"
OUT.mkdir(parents=True, exist_ok=True)

# (RP_ID, arxiv_id, first_author, year, short_title) — for filename
PAPERS = [
    ("RP08", "2207.00032", "Aminabadi", 2022, "DeepSpeed-Inference"),
    ("RP11", "2208.07339", "Dettmers",  2022, "LLM-int8"),
    ("RP14", "2206.01861", "Yao",       2022, "ZeroQuant"),
    ("RP19", "2310.01382", "Jaiswal",   2023, "CompressingLLMs"),
    ("RP23", "2306.11698", "Wang",      2023, "DecodingTrust"),
    ("RP25", "2309.06180", "Kwon",      2023, "PagedAttention"),
    ("RP31", "2306.14048", "Zhang",     2023, "H2O"),
    ("RP33", "2303.06865", "Sheng",     2023, "FlexGen"),
    ("RP37", "2312.11514", "Alizadeh",  2023, "LLMinaFlash"),
    ("RP39", "2310.01801", "Ge",        2023, "FastGen"),
    ("RP40", "2305.14314", "Dettmers",  2023, "QLoRA"),
    ("RP41", "2407.04965", "Xu",        2024, "BeyondPerplexity"),
    ("RP42", "2406.01721", "Lin",       2024, "DuQuant"),
    ("RP43", "2409.11055", "Lee",       2024, "EdgeToGiant"),
    ("RP44", "2402.04249", "Mazeika",   2024, "HarmBench"),
    ("RP46", "2401.05561", "Sun",       2024, "TrustLLM"),
    ("RP47", "2506.04645", "Erdil",     2025, "InferenceEconomics"),
    ("RP50", "2502.15799", "Kharinaev", 2025, "OpenMiniSafety"),
]

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 AI503-SLR-fetch"


def fetch(rp, arxiv_id, author, year, short):
    fname = f"{rp}_{author}_{year}.pdf"
    out = OUT / fname
    if out.exists() and out.stat().st_size > 1000:
        return ("skip", fname, out.stat().st_size)
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        out.write_bytes(data)
        return ("ok", fname, len(data))
    except Exception as e:
        return ("fail", fname, str(e))


def main():
    results = []
    for rp, ax, author, yr, short in PAPERS:
        r = fetch(rp, ax, author, yr, short)
        results.append((rp, ax, *r))
        status_label = {"ok": "[OK] downloaded", "skip": "[--] skipped (exists)", "fail": "[XX] failed"}[r[0]]
        size = r[2] if isinstance(r[2], int) else "—"
        print(f"  {rp}  {ax:13s}  {status_label:<20s}  {size:>10}  {r[1]}")
        time.sleep(1.5)  # be polite to arxiv.org
    ok = sum(1 for _, _, s, *_ in results if s == "ok")
    skip = sum(1 for _, _, s, *_ in results if s == "skip")
    fail = sum(1 for _, _, s, *_ in results if s == "fail")
    print()
    print(f"Total: {len(results)} papers — {ok} downloaded, {skip} skipped, {fail} failed")
    if fail:
        print("\nFailures:")
        for rp, ax, st, fname, err in [r for r in results if r[2] == "fail"]:
            print(f"  {rp}  {ax}  -> {err}")


if __name__ == "__main__":
    main()
