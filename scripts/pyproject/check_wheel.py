#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def distribution_version(wheel_path: Path) -> str:
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_names = [
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(f"expected one METADATA file, got {metadata_names}")
        metadata = wheel.read(metadata_names[0]).decode()

    versions = [
        line.removeprefix("Version: ")
        for line in metadata.splitlines()
        if line.startswith("Version: ")
    ]
    if len(versions) != 1:
        raise RuntimeError(f"expected one package version, got {versions}")
    return versions[0]


def check_version(version: str) -> None:
    release_tag = os.environ.get("RELEASE_TAG", "")
    if release_tag:
        expected = release_tag.removeprefix("xtc-v")
        if version != expected:
            raise RuntimeError(
                f"tag {release_tag} must build version {expected}, got {version}"
            )
    elif ".dev" not in version:
        expected_tag = f"xtc-v{version}"
        tags = subprocess.run(
            ["git", "tag", "--points-at", "HEAD", "--list", expected_tag],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if expected_tag not in tags:
            raise RuntimeError(
                "development build must have a dev version or match an exact "
                f"tag on the current commit, got {version}"
            )


def check_installation(wheel_path: Path, version: str) -> None:
    with tempfile.TemporaryDirectory(prefix="xtc-wheel-test-") as directory:
        venv = Path(directory)
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel_path)],
            check=True,
        )
        installed = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import importlib.metadata, xtc; "
                    "installed = importlib.metadata.version('xtc-tools'); "
                    "assert xtc.__version__ == installed; print(installed)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if installed != version:
            raise RuntimeError(
                f"built version {version}, installed version {installed}"
            )


def main() -> None:
    dist = Path("dist")
    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError(
            f"expected one wheel and one sdist, got {wheels=} and {sdists=}"
        )

    version = distribution_version(wheels[0])
    check_version(version)
    check_installation(wheels[0].resolve(), version)
    print(f"Checked xtc-tools {version} (import package: xtc)")


if __name__ == "__main__":
    main()
