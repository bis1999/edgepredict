# src/linkpredx/git_link_script/link_prediction/models/downloader.py
from __future__ import annotations
import os, hashlib
from pathlib import Path
from typing import Dict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Release config ---
TAG   = "v0.1.0"
OWNER = "bis1999"
REPO  = "edgepredict"
BASE  = f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}"
FILES: Dict[str, str] = {
    "auc_label_encoder.pkl":        "41be0890f06d446db2085d7cde664ff1b47ca8e87985b34f9a7e06af41a0eb34",
    "auc_score_regressor.pkl":      "2bdb2f204e371d748aeea85a3a8c046afe914b3917b98e870b4fe1a4d7bf99fe",
    "best_auc_model_classifier.pkl":"28013ab22d9111e24180ab17d9176ba661cf67ac566a8c820ee3dbb8baba506d",
    "best_topk_model_classifier.pkl":"36b54a5e171991e1b1b312427bdaefc000c90bd5387a6e4af4fffc7a50840ada",
    "topk_label_encoder.pkl":       "178ce76129e643263c22faee77f675f0941ff10c119eb41b74e74a017c9b8616",
    "topk_score_regressor.pkl":     "9fe6ddd7f36a95e3e54a5a2f887494560a744213e6d2c8f2e8da6f1a54e799f8",
}
CHUNK = 4 * 1024 * 1024  # 4MB

def _sha256(path: Path, buf=1<<20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()

def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=8, backoff_factor=0.6,
                    status_forcelist=[429,500,502,503,504],
                    allowed_methods=["GET","HEAD"])
    s.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=8, pool_maxsize=8))
    headers = {"User-Agent": "edgepredict-downloader/1.0", "Accept": "application/octet-stream"}
    if (tok := os.getenv("GITHUB_TOKEN")):
        headers["Authorization"] = f"Bearer {tok}"
    s.headers.update(headers)
    return s

def _download_one(name: str, dest: Path) -> Path:
    url = f"{BASE}/{name}"
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / name
    tmp = out.with_suffix(out.suffix + ".part")

    s = _session()
    try:
        head = s.head(url, timeout=30); head.raise_for_status()
        total = int(head.headers.get("Content-Length") or 0)
    except Exception:
        total = 0

    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else {}

    with s.get(url, stream=True, headers=headers, timeout=60) as r:
        if r.status_code not in (200, 206):
            r.raise_for_status()
        mode = "ab" if resume_from > 0 else "wb"
        done = resume_from
        with open(tmp, mode) as f:
            for chunk in r.iter_content(CHUNK):
                if chunk:
                    f.write(chunk)
                    if total:
                        done += len(chunk)
                        pct = done * 100 // total
                        print(f"\r{name}: {pct:3d}% ({done//1024//1024}/{total//1024//1024} MB)", end="")
        if total: print()
    tmp.replace(out)
    return out

def _default_models_dir() -> Path:
    """
    Default to the **git_link_script/meta_learner_models** folder.
    """
    here = Path(__file__).resolve()
    git_link_script = here.parents[2]              # .../git_link_script
    return git_link_script / "meta_learner_models"

def ensure_models(dest: Path | str | None = None) -> Path:
    """
    Ensure all release assets are present & verified in the target directory.
    If dest is None, use <git_link_script>/meta_learner_models.
    Returns the destination Path.
    """
    dest_path = Path(dest).expanduser().resolve() if dest else _default_models_dir()
    for name, want in FILES.items():
        out = dest_path / name
        if out.exists():
            try:
                if _sha256(out) == want:
                    print(f"✓ {name} already verified"); continue
                else:
                    print(f"↻ {name} checksum mismatch; re-downloading")
                    out.unlink(missing_ok=True)
            except Exception:
                print(f"↻ {name} unreadable; re-downloading")
        print(f"↓ {name}")
        out = _download_one(name, dest_path)
        got = _sha256(out)
        if got != want:
            out.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum FAILED for {name}\n  got:  {got}\n  want: {want}")
        print(f"✓ {name} verified ({out.stat().st_size//1024//1024} MB)")
    print(f"\nAll models are ready in {dest_path}")
    return dest_path
