#!/usr/bin/env python3
"""Download closed-object example point clouds.

This intentionally uses closed meshes / point clouds (bunny, dragon, teapot)
instead of smooth open surfaces so downstream sampling assumptions hold better.
"""
from __future__ import annotations

import argparse
import pathlib
import urllib.request

SOURCES = {
    "bunny": "https://github.com/alecjacobson/common-3d-test-models/raw/master/data/stanford-bunny.obj",
    "dragon": "https://github.com/alecjacobson/common-3d-test-models/raw/master/data/xyzrgb_dragon.ply",
    "teapot": "https://github.com/alecjacobson/common-3d-test-models/raw/master/data/teapot.obj",
}


def download(url: str, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download closed-object example point clouds.")
    parser.add_argument("--out-dir", default="assets/point_clouds", help="Output directory for downloaded files")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    for name, url in SOURCES.items():
        ext = pathlib.Path(url).suffix or ".ply"
        destination = out_dir / f"{name}{ext}"
        print(f"Downloading {name} -> {destination}")
        download(url, destination)

    print("Done. Downloaded bunny, dragon, and teapot closed-object samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
