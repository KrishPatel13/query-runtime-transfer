import importlib.util
import io
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fix = _load("tpcds_fix_utf8")
rec = _load("tpcds_load_record")


def test_latin1_bytes_become_utf8():
    assert fix.fix_line(b"1|C\xd4TE D'IVOIRE|\n") == "1|CÔTE D'IVOIRE|\n".encode("utf-8")


def test_valid_utf8_passes_unchanged():
    raw = "2|RÉUNION|\n".encode("utf-8")
    assert fix.fix_line(raw) == raw


def test_stream_counts_changed_lines_and_keeps_fields():
    dst = io.BytesIO()
    assert fix.fix_stream(io.BytesIO(b"a|1|\nC\xd4TE|2|\nb|3|\n"), dst) == (3, 1)
    out = dst.getvalue().decode("utf-8")
    assert out == "a|1|\nCÔTE|2|\nb|3|\n"
    assert [line.count("|") for line in out.splitlines()] == [2, 2, 2]


def test_name_value_arguments():
    assert rec.pairs(["dsdgen=43.7", "load=80.5"]) == {"dsdgen": 43.7, "load": 80.5}
    assert rec.pairs(["customer=3"], int) == {"customer": 3}
    with pytest.raises(SystemExit):
        rec.pairs(["dsdgen"])


def test_macros_leave_out_the_version_table():
    record = {"scale_factor": 1.0, "flat_file_bytes": 1_253_241_614, "database_bytes": 3_142_981_311,
              "foreign_keys": 102, "indexes": 120, "seconds": {"dsdgen": 43.7, "load": 80.5},
              "tables": {"dbgen_version": {"rows": 1}, "store_sales": {"rows": 2_880_404},
                         "item": {"rows": 18_000}}}
    m = rec.macros(record)
    assert m["DSTables"] == "2"
    assert m["DSRowsMillions"] == "2.9"
    assert (m["DSDatabaseGB"], m["DSFlatFileGB"]) == ("3.1", "1.3")
    assert (m["DSDsdgenSeconds"], m["DSLoadMinutes"]) == ("44", "1.3")
    assert all(name.isalpha() for name in m)    # LaTeX macro names are letters only
