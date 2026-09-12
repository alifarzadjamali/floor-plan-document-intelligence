from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD = "https://zenodo.org/api/records/2613548"
CHUNK_SIZE = 8 * 1024 * 1024


def metadata() -> dict[str, object]:
    with urllib.request.urlopen(ZENODO_RECORD) as response:  # noqa: S310 (fixed HTTPS URL)
        return json.load(response)


def download(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    if existing == expected_size:
        return
    request = urllib.request.Request(url, headers={"Range": f"bytes={existing}-"})
    with urllib.request.urlopen(request) as response:  # noqa: S310 (Zenodo metadata URL)
        mode = "ab" if existing and response.status == 206 else "wb"
        with destination.open(mode) as target:
            shutil.copyfileobj(response, target, CHUNK_SIZE)
    if destination.stat().st_size != expected_size:
        raise RuntimeError("Downloaded archive size does not match Zenodo metadata")


def checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(resolved_destination):
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CubiCasa5K from its Zenodo record")
    parser.add_argument("--archive", type=Path, default=Path("data/cubicasa5k.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/cubicasa5k"))
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    record = metadata()
    files = [item for item in record["files"] if item["key"] == "cubicasa5k.zip"]
    if len(files) != 1:
        raise RuntimeError("Zenodo record does not expose exactly one cubicasa5k.zip")
    item = files[0]
    download(item["links"]["self"], args.archive, item["size"])
    algorithm, expected = item["checksum"].split(":", 1)
    actual = checksum(args.archive, algorithm)
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch: expected {expected}, got {actual}")
    safe_extract(args.archive, args.output)
    if not args.keep_archive:
        args.archive.unlink()
    print(f"CubiCasa5K ready at {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
