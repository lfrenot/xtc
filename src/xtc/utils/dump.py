#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import sys
import yaml
import json
import csv
from typing import TextIO, Any


def dump_plain(
    payload: Any,
    key: str | None = None,
    file: TextIO = sys.stdout,
    format: str = "yaml",
):
    """
    Dump a plain object (ditc|list|scalar) to the given file/format.

    If key is specified, the payload must a be dict and the dumped
    object is the value for this key.
    """
    if key is not None:
        assert isinstance(payload, dict)
        dump_plain(payload[key], file=file, format=format)
        return
    match format:
        case "yaml":
            yaml.safe_dump(
                payload,
                file,
                sort_keys=False,
                default_flow_style=False,
            )
        case "python":
            print(str(payload), file=file, flush=True)
        case "json":
            json.dump(payload, file, sort_keys=False)
            file.write("\n")
        case "jsonl":
            if not isinstance(payload, list):
                payload = [payload]
            for row in payload:
                json.dump(row, file, sort_keys=False)
                file.write("\n")
        case "csv":
            if not isinstance(payload, list):
                payload = [payload]
            _dump_csv(payload, file)
        case _:
            raise ValueError(f"output format {format} not supported")


def _flatten_plain(payload: Any) -> tuple[list[str], list[dict[str, Any]]]:
    def _flatten(key: str, payload: Any, obj: dict):
        _join = lambda k: f"{key}.{k}" if key else k
        if isinstance(payload, dict):
            for k, v in payload.items():
                _flatten(_join(k), v, obj)
        elif isinstance(payload, list):
            for i, v in enumerate(payload):
                _flatten(_join(str(i)), v, obj)
        else:
            if key in obj:
                raise ValueError(f"key ambiguity when processing: {key}: {payload}")
            obj[key] = payload

    assert isinstance(payload, list)
    header = []
    rows = []
    for row in payload:
        obj = {}
        _flatten("", row, obj)
        header = list({**dict.fromkeys(header), **dict.fromkeys(obj)})
        rows.append(obj)
    return header, rows


def _dump_csv(
    payload: Any,
    file: TextIO = sys.stdout,
):
    assert isinstance(payload, list)
    writer = csv.writer(file)
    header, rows = _flatten_plain(payload)
    writer.writerow(header)
    for obj in rows:
        row = [obj.get(name, "") for name in header]
        writer.writerow(row)
