#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from pathlib import Path
import hashlib
import tempfile
import yaml


def get_blob_digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def get_str_digest(data: str) -> str:
    return get_blob_digest(data.encode())


def get_dict_digest(data: dict) -> str:
    dict_str = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
    )
    digest = get_str_digest(dict_str)
    return digest


def get_digest_components(digest: str) -> tuple[str, ...]:
    return digest[:2], digest[2:4], digest[4:]


def save_blob(blob_dir: Path | str, blob: bytes) -> str:
    blob_dir = Path(blob_dir)
    digest = get_blob_digest(blob)
    components = get_digest_components(digest)
    blob_dir = blob_dir / Path(*components[:-1])
    blob_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=blob_dir, suffix=".tmp", delete=False) as tmp:
        tmp.write(blob)
    Path(tmp.name).replace(blob_dir / components[-1])
    return digest


def load_blob(blob_dir: Path | str, digest: str) -> bytes:
    blob_dir = Path(blob_dir)
    components = get_digest_components(digest)
    blob_path = blob_dir / Path(*components)
    with open(blob_path, "rb") as inf:
        blob = inf.read()
    return blob


def load_blob_str(blob_dir: Path | str, digest: str) -> str:
    blob = load_blob(blob_dir, digest)
    return blob.decode()


def save_blob_str(blob_dir: Path | str, data: str) -> str:
    return save_blob(blob_dir, data.encode())


def save_blob_dict(blob_dir: Path | str, data: dict) -> str:
    dict_str = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
    )
    digest = save_blob_str(blob_dir, dict_str)
    return digest


def load_blob_dict(blob_dir: Path | str, digest: str) -> dict:
    blob_str = load_blob_str(blob_dir, digest)
    data = yaml.safe_load(blob_str)
    return data
