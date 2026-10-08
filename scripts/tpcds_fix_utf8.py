"""Copy a dsdgen data file from stdin to stdout as valid UTF-8.

COPY into a UTF-8 database rejects any byte sequence that is not valid UTF-8.
Some dsdgen builds write a few names from their word lists in Latin-1, for
example the Ô in CÔTE D'IVOIRE as the single byte 0xD4, where UTF-8 needs
the two bytes 0xC3 0x94. This filter reads every byte that is not part of a
valid UTF-8 sequence as the Latin-1 character with that code, so such names
come out as the intended letters, and passes valid UTF-8 through unchanged.
Nothing is dropped, so the number of fields per row stays the same.

scripts/tpcds_load.sh runs a file through this filter only when iconv finds
invalid bytes in it, and stores the number of changed lines (written to
--count-file) in the load record. The dsdgen built from gregrahn/tpcds-kit
writes these names as valid UTF-8, so at SF=1 no file has needed it; the
filter stays for other builds of the tools.

Usage: python3 scripts/tpcds_fix_utf8.py --count-file N.txt < in.dat > out.dat
"""
from __future__ import annotations

import argparse
import codecs
import sys
from typing import BinaryIO


def _as_latin1(err: UnicodeError) -> tuple[str, int]:
    if not isinstance(err, UnicodeDecodeError):
        raise err
    return err.object[err.start:err.end].decode("latin-1"), err.end


codecs.register_error("cs680_latin1", _as_latin1)


def fix_line(raw: bytes) -> bytes:
    """The line as valid UTF-8, with invalid bytes read as Latin-1."""
    return raw.decode("utf-8", "cs680_latin1").encode("utf-8")


def fix_stream(src: BinaryIO, dst: BinaryIO) -> tuple[int, int]:
    """Copy src to dst line by line; return (lines, changed lines)."""
    lines = changed = 0
    for raw in src:
        fixed = fix_line(raw)
        lines += 1
        changed += fixed != raw
        dst.write(fixed)
    return lines, changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count-file", default=None, help="write the number of changed lines here")
    args = ap.parse_args()
    lines, changed = fix_stream(sys.stdin.buffer, sys.stdout.buffer)
    sys.stdout.buffer.flush()
    if args.count_file:
        with open(args.count_file, "w", encoding="utf-8") as f:
            f.write(f"{changed}\n")
    print(f"re-encoded {changed} of {lines} lines as UTF-8", file=sys.stderr)


if __name__ == "__main__":
    main()
