import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "generate_qgen_queries", Path(__file__).resolve().parents[1] / "scripts" / "generate_qgen_queries.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

# Shapes copied from real qgen 2.17.3 output built with DATABASE=POSTGRESQL.
Q1 = """-- using 101 as a seed to the RNG


select
\tl_returnflag,
\tcount(*) as count_order
from
\tlineitem
where
\tl_shipdate <= date '1998-12-01' - interval '77' day (3)
group by
\tl_returnflag
order by
\tl_returnflag;
limit -1;
"""

Q3 = """-- using 101 as a seed to the RNG


select
\tl_orderkey
from
\tcustomer
order by
\trevenue desc,
\to_orderdate;
limit 10;
"""

Q15 = """-- using 101 as a seed to the RNG

create view revenue0 (supplier_no, total_revenue) as
\tselect
\t\tl_suppkey,
\t\tsum(l_extendedprice * (1 - l_discount))
\tfrom
\t\tlineitem
\tgroup by
\t\tl_suppkey;


select
\ts_suppkey,
\ttotal_revenue
from
\tsupplier,
\trevenue0
where
\ts_suppkey = supplier_no
\tand total_revenue = (
\t\tselect
\t\t\tmax(total_revenue)
\t\tfrom
\t\t\trevenue0
\t)
order by
\ts_suppkey;

drop view revenue0;
limit -1;
"""


def test_no_limit_and_interval_precision():
    sql = gen.to_postgres(Q1)
    assert "day (3)" not in sql
    assert "interval '77' day" in sql
    assert "limit" not in sql
    assert sql.count(";") == 1 and sql.rstrip().endswith(";")
    assert "--" not in sql


def test_row_limit_moves_inside_statement():
    sql = gen.to_postgres(Q3)
    assert sql.rstrip().endswith("o_orderdate\nlimit 10;")
    assert sql.count(";") == 1


def test_view_becomes_with_clause():
    sql = gen.to_postgres(Q15)
    assert sql.startswith("with revenue0 (supplier_no, total_revenue) as (")
    assert "create view" not in sql and "drop view" not in sql
    assert sql.count(";") == 1


def test_rejects_leftover_statements():
    with pytest.raises(ValueError):
        gen.to_postgres("select 1; select 2;\nlimit -1;\n")


def test_template_ranges():
    assert gen.parse_templates("1-3,6,14") == [1, 2, 3, 6, 14]


def test_reordered_parameters_are_duplicates():
    # q12_003 and q12_011 in the first pilot workload differed only like this.
    a = "select 1 from lineitem where l_shipmode in ('TRUCK', 'RAIL') and l_receiptdate >= date '1996-01-01';"
    b = "select 1 from lineitem where l_shipmode in ('RAIL', 'TRUCK') and l_receiptdate >= date '1996-01-01';"
    c = "select 1 from lineitem where l_shipmode in ('RAIL', 'SHIP') and l_receiptdate >= date '1996-01-01';"
    assert gen.dedupe_key(a) == gen.dedupe_key(b)
    assert gen.dedupe_key(a) != gen.dedupe_key(c)


def test_decimal_parameters_distinguish_queries():
    a = "select 1 where l_discount between .06 - 0.01 and .06 + 0.01 and l_quantity < 24;"
    b = "select 1 where l_discount between .07 - 0.01 and .07 + 0.01 and l_quantity < 24;"
    assert gen.dedupe_key(a) != gen.dedupe_key(b)
