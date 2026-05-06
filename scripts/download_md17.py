from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jepa_md17_mlff.data import load_md17_npz, write_metadata


ETHANOL_URLS = [
    "http://www.quantum-machine.org/gdml/data/npz/md17_ethanol.npz",
    "https://www.quantum-machine.org/gdml/data/npz/md17_ethanol.npz",
    "http://www.quantum-machine.org/gdml/data/npz/ethanol_dft.npz",
    "https://www.quantum-machine.org/gdml/data/npz/ethanol_dft.npz",
]


def download(url: str, out_path: Path, timeout: int = 60) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0"))
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        with tmp_path.open("wb") as handle:
            with tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        bar.update(len(chunk))
        tmp_path.replace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate MD17 ethanol data.")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "md17_ethanol.npz")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data" / "md17_ethanol_metadata.json")
    parser.add_argument("--url", action="append", default=[], help="Extra URL to try before defaults.")
    parser.add_argument("--force", action="store_true", help="Re-download even if output exists.")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out.exists() or args.force:
        errors = []
        for url in [*args.url, *ETHANOL_URLS]:
            try:
                print(f"Trying {url}")
                download(url, args.out)
                break
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                if args.out.exists():
                    args.out.unlink()
        else:
            joined = "\n".join(errors)
            raise SystemExit(
                "Could not download ethanol data from known MD17 URLs.\n"
                "You can manually place an MD17/rMD17-compatible .npz at "
                f"{args.out} or pass --url.\n\nAttempts:\n{joined}"
            )
    else:
        print(f"Using existing dataset: {args.out}")

    arrays = load_md17_npz(args.out)
    meta = write_metadata(args.out, args.metadata)
    print(
        f"Validated {args.out.name}: frames={arrays.R.shape[0]}, atoms={arrays.R.shape[1]}, "
        f"metadata={args.metadata}"
    )
    print(meta)


if __name__ == "__main__":
    main()
