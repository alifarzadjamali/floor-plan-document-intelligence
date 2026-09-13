from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

VERSION = "5.5.3.20260724"
URL = (
    "https://github.com/tesseract-ocr/tesseract/releases/download/5.5.3/"
    f"tesseract-ocr-w64-setup-{VERSION}.exe"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install Tesseract 5 locally inside the project virtual environment"
    )
    parser.add_argument("--venv", type=Path, default=Path(".venv"))
    args = parser.parse_args()
    destination = args.venv.resolve() / "Tesseract-OCR"
    executable = destination / "tesseract.exe"
    if executable.is_file():
        print(executable)
        return 0
    installer = args.venv.resolve() / f"tesseract-ocr-w64-setup-{VERSION}.exe"
    with urllib.request.urlopen(URL) as response, installer.open("wb") as stream:  # noqa: S310 (fixed URL)
        stream.write(response.read())
    subprocess.run([str(installer), "/S", f"/D={destination}"], check=True)
    installer.unlink(missing_ok=True)
    if not executable.is_file():
        raise RuntimeError("Tesseract installer completed but tesseract.exe was not created")
    print(executable)
    return 0


if __name__ == "__main__":
    sys.exit(main())
