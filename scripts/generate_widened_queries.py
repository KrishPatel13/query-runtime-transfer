"""Build the widened TPC-H workload (queries/tpch_sf1_v2) and its fixed pool/test split.

The pilot's 420 qgen queries (queries/tpch_sf1) vary runtime very little
within a template (WithinMax in results/pilot/numbers.tex), because the TPC-H
specification draws the parameters from narrow ranges. The full study keeps
those 420 queries unchanged, as variant "std", and adds 20 queries for each
of the 13 templates with a range parameter, as variant "wide", with values
drawn from ranges well beyond the specification: 680 queries in all.

Method
  * Each template text below is the TPC-H kit's query (TPC-H tools 2.17.3,
    dbgen/queries) as qgen writes it for PostgreSQL: the header comments and
    the :x and :o lines are dropped, ":n N" becomes the line "limit N;", and
    Q15's view is named revenue0 (stream 0). The kit's markers (:1, :2, ...)
    became named placeholders, and so did the fixed literals that the
    widening changes: the window lengths, the Q6 discount band and the Q19
    quantity band and size limits. A query is its template with the values
    filled in, rewritten by to_postgres() from generate_qgen_queries.py like
    every pilot query.
  * Every pilot query of these templates is checked to be its template filled
    with its own values, byte for byte, and the manifest records those values.
    The template texts are therefore the kit's, and a standard and a widened
    query of one template differ only in their values.
  * Each template draws from its own seeded random stream. A draw is rejected
    and drawn again when its sorted literal values (dedupe_key) equal those of
    a standard or an earlier widened query of the template, or when it
    filters the same date window as one of them written in other units (3
    months and 92 days from the same first day) with the same other values in
    any order.
  * Date windows: the length is log-uniform, so that short and long windows
    are about equally common on the log scale on which the models predict
    runtime, and the first day is uniform over the days that keep the whole
    window inside the column's data range (o_orderdate 1992-01-01 to
    1998-08-02, l_shipdate 1992-01-02 to 1998-12-01, l_receiptdate 1992-01-04
    to 1998-12-31). Widened windows are written in days.

Ranges, standard (qgen, TPC-H rules) -> widened. Draws are uniform unless
marked log-uniform. The market segment, region, nations, ship modes and
brands are drawn uniformly from the lists in qgen's dists.dss, which the
standard queries also use.
  Q1   days before 1998-12-01: 60 to 120 -> 0 to 2400
  Q3   date: 1995-03-01 to 1995-03-31 -> 1992-01-01 to 1998-08-02
  Q4   o_orderdate window: 3 months from the first of a month, 1993-01 to
       1997-10 -> 7 to 1461 days (log-uniform) anywhere in the column's range
  Q5   o_orderdate window: 1 year from January 1, 1993 to 1997 -> as Q4
  Q6   l_shipdate window: 1 year from January 1, 1993 to 1997 -> as Q4;
       discount center 0.02 to 0.09 -> 0.00 to 0.10 and discount band 0.01
       -> 0.00 to 0.05, both in steps of 0.01; quantity limit 24 to 25 -> 2
       to 51
  Q10  o_orderdate window: 3 months from the first of a month, 1993-02 to
       1995-01 -> as Q4. No order placed after mid-1995 has a returned item,
       so a window there returns nothing, but the query still reads and joins
       the window's orders.
  Q11  fraction of the nation's total stock value: 0.0001 -> 0.0000001 to
       0.001, log-uniform (from almost every part listed to none)
  Q12  l_receiptdate window: 1 year from January 1, 1993 to 1997 -> as Q4
  Q14  l_shipdate window: 1 month from the first of a month, 1993-01 to
       1997-12 -> as Q4
  Q15  l_shipdate window: 3 months from the first of a month, 1993-01 to
       1997-10 -> as Q4
  Q18  order quantity threshold: 312 to 315 -> 30 to 311
  Q19  quantity lower bounds 1 to 10, 10 to 20 and 20 to 30 -> 1 to 50 each;
       quantity band 10 -> 0 to 49; size limits 5, 10 and 15 -> 1 to 50
       each, log-uniform
  Q20  part name prefix: one color word -> the first 1 to all characters of
       two different color words joined by a space (a prefix that ends at
       the space is the first word); a shorter prefix matches more parts,
       one that reaches into the second word far fewer. l_shipdate window:
       1 year from January 1, 1993 to 1997 -> as Q4

Runtime drivers found on M1 before choosing these ranges (one run each,
2026-09-30): the date windows, Q1's cutoff and Q3's date set how many rows
are read and joined; Q18's threshold sets how many orders reach the outer
join, so runtime rises steadily as it falls; Q19's size limits and
Q20's prefix set how many parts are looked up. Q11's fraction only decides
how many of the aggregated parts are kept and sorted, so its runtime changes
much less than the others' (the section Q11 of results/workload_check/README.md
measures how much), and PostgreSQL's estimates do not depend on the Q11
fraction or the Q18 threshold at all.

Split: split.csv puts every query in the pool (70%) or the test set (30%),
once, inside every (template, variant) group, with the rounding of
scripts/pilot_adaptation.py: a group of n queries gives max(1, round(0.3 n))
test queries, and a group with a single query goes to the pool. Models train
only on pool queries and are scored only on test queries, and the same IDs
are used on every machine (FINAL_PROPOSAL_PLAN.md, section 4).

Writes queries/tpch_sf1_v2/*.sql (standard queries copied byte for byte,
widened ones named q01_w01 to q20_w20), manifest.json (query_id, template,
variant, parameters; a standard query also keeps its qgen seed) and split.csv
(query_id, template, variant, set). scripts/check_workload.py plans every
query and times the widened ones on M1 (results/workload_check).

Usage: python scripts/generate_widened_queries.py [--force]
"""
from __future__ import annotations

import argparse
import calendar
import csv
import datetime as dt
import json
import math
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from generate_qgen_queries import dedupe_key, to_postgres  # noqa: E402

STD_DIR = HERE / "queries/tpch_sf1"
OUT_DIR = HERE / "queries/tpch_sf1_v2"
SEED = 680
PER_TEMPLATE = 20
TEST_FRAC = 0.3
MAX_TRIES = 2000

# Data range of each date column a window filters on (TPC-H SF1, checked on M1).
ORDER_DATES = (dt.date(1992, 1, 1), dt.date(1998, 8, 2))       # o_orderdate
SHIP_DATES = (dt.date(1992, 1, 2), dt.date(1998, 12, 1))       # l_shipdate
RECEIPT_DATES = (dt.date(1992, 1, 4), dt.date(1998, 12, 31))   # l_receiptdate
SHORTEST, LONGEST = 7, 1461    # widened window length in days: one week to four years

# The value lists qgen draws from (dists.dss), used unchanged.
SEGMENTS = ["AUTOMOBILE", "BUILDING", "FURNITURE", "HOUSEHOLD", "MACHINERY"]
REGIONS = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
NATIONS = ["ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "EGYPT", "ETHIOPIA", "FRANCE", "GERMANY", "INDIA",
           "INDONESIA", "IRAN", "IRAQ", "JAPAN", "JORDAN", "KENYA", "MOROCCO", "MOZAMBIQUE", "PERU", "CHINA",
           "ROMANIA", "SAUDI ARABIA", "VIETNAM", "RUSSIA", "UNITED KINGDOM", "UNITED STATES"]
SHIP_MODES = ["REG AIR", "AIR", "RAIL", "TRUCK", "MAIL", "FOB", "SHIP"]
COLORS = ["almond", "antique", "aquamarine", "azure", "beige", "bisque", "black", "blanched", "blue", "blush",
          "brown", "burlywood", "burnished", "chartreuse", "chiffon", "chocolate", "coral", "cornflower",
          "cornsilk", "cream", "cyan", "dark", "deep", "dim", "dodger", "drab", "firebrick", "floral", "forest",
          "frosted", "gainsboro", "ghost", "goldenrod", "green", "grey", "honeydew", "hot", "indian", "ivory",
          "khaki", "lace", "lavender", "lawn", "lemon", "light", "lime", "linen", "magenta", "maroon", "medium",
          "metallic", "midnight", "mint", "misty", "moccasin", "navajo", "navy", "olive", "orange", "orchid",
          "pale", "papaya", "peach", "peru", "pink", "plum", "powder", "puff", "purple", "red", "rose", "rosy",
          "royal", "saddle", "salmon", "sandy", "seashell", "sienna", "sky", "slate", "smoke", "snow", "spring",
          "steel", "tan", "thistle", "tomato", "turquoise", "violet", "wheat", "white", "yellow"]

# The kit's query texts as qgen writes them for PostgreSQL, with named placeholders.
TEMPLATES: dict[int, str] = {
    1: """\
select
\tl_returnflag,
\tl_linestatus,
\tsum(l_quantity) as sum_qty,
\tsum(l_extendedprice) as sum_base_price,
\tsum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
\tsum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
\tavg(l_quantity) as avg_qty,
\tavg(l_extendedprice) as avg_price,
\tavg(l_discount) as avg_disc,
\tcount(*) as count_order
from
\tlineitem
where
\tl_shipdate <= date '1998-12-01' - interval '{delta}' day (3)
group by
\tl_returnflag,
\tl_linestatus
order by
\tl_returnflag,
\tl_linestatus;
limit -1;
""",
    3: """\
select
\tl_orderkey,
\tsum(l_extendedprice * (1 - l_discount)) as revenue,
\to_orderdate,
\to_shippriority
from
\tcustomer,
\torders,
\tlineitem
where
\tc_mktsegment = '{segment}'
\tand c_custkey = o_custkey
\tand l_orderkey = o_orderkey
\tand o_orderdate < date '{date}'
\tand l_shipdate > date '{date}'
group by
\tl_orderkey,
\to_orderdate,
\to_shippriority
order by
\trevenue desc,
\to_orderdate;
limit 10;
""",
    4: """\
select
\to_orderpriority,
\tcount(*) as order_count
from
\torders
where
\to_orderdate >= date '{start}'
\tand o_orderdate < date '{start}' + interval '{window}' {unit}
\tand exists (
\t\tselect
\t\t\t*
\t\tfrom
\t\t\tlineitem
\t\twhere
\t\t\tl_orderkey = o_orderkey
\t\t\tand l_commitdate < l_receiptdate
\t)
group by
\to_orderpriority
order by
\to_orderpriority;
limit -1;
""",
    5: """\
select
\tn_name,
\tsum(l_extendedprice * (1 - l_discount)) as revenue
from
\tcustomer,
\torders,
\tlineitem,
\tsupplier,
\tnation,
\tregion
where
\tc_custkey = o_custkey
\tand l_orderkey = o_orderkey
\tand l_suppkey = s_suppkey
\tand c_nationkey = s_nationkey
\tand s_nationkey = n_nationkey
\tand n_regionkey = r_regionkey
\tand r_name = '{region}'
\tand o_orderdate >= date '{start}'
\tand o_orderdate < date '{start}' + interval '{window}' {unit}
group by
\tn_name
order by
\trevenue desc;
limit -1;
""",
    6: """\
select
\tsum(l_extendedprice * l_discount) as revenue
from
\tlineitem
where
\tl_shipdate >= date '{start}'
\tand l_shipdate < date '{start}' + interval '{window}' {unit}
\tand l_discount between {discount} - {discount_band} and {discount} + {discount_band}
\tand l_quantity < {quantity};
limit -1;
""",
    10: """\
select
\tc_custkey,
\tc_name,
\tsum(l_extendedprice * (1 - l_discount)) as revenue,
\tc_acctbal,
\tn_name,
\tc_address,
\tc_phone,
\tc_comment
from
\tcustomer,
\torders,
\tlineitem,
\tnation
where
\tc_custkey = o_custkey
\tand l_orderkey = o_orderkey
\tand o_orderdate >= date '{start}'
\tand o_orderdate < date '{start}' + interval '{window}' {unit}
\tand l_returnflag = 'R'
\tand c_nationkey = n_nationkey
group by
\tc_custkey,
\tc_name,
\tc_acctbal,
\tc_phone,
\tn_name,
\tc_address,
\tc_comment
order by
\trevenue desc;
limit 20;
""",
    11: """\
select
\tps_partkey,
\tsum(ps_supplycost * ps_availqty) as value
from
\tpartsupp,
\tsupplier,
\tnation
where
\tps_suppkey = s_suppkey
\tand s_nationkey = n_nationkey
\tand n_name = '{nation}'
group by
\tps_partkey having
\t\tsum(ps_supplycost * ps_availqty) > (
\t\t\tselect
\t\t\t\tsum(ps_supplycost * ps_availqty) * {fraction}
\t\t\tfrom
\t\t\t\tpartsupp,
\t\t\t\tsupplier,
\t\t\t\tnation
\t\t\twhere
\t\t\t\tps_suppkey = s_suppkey
\t\t\t\tand s_nationkey = n_nationkey
\t\t\t\tand n_name = '{nation}'
\t\t)
order by
\tvalue desc;
limit -1;
""",
    12: """\
select
\tl_shipmode,
\tsum(case
\t\twhen o_orderpriority = '1-URGENT'
\t\t\tor o_orderpriority = '2-HIGH'
\t\t\tthen 1
\t\telse 0
\tend) as high_line_count,
\tsum(case
\t\twhen o_orderpriority <> '1-URGENT'
\t\t\tand o_orderpriority <> '2-HIGH'
\t\t\tthen 1
\t\telse 0
\tend) as low_line_count
from
\torders,
\tlineitem
where
\to_orderkey = l_orderkey
\tand l_shipmode in ('{shipmode1}', '{shipmode2}')
\tand l_commitdate < l_receiptdate
\tand l_shipdate < l_commitdate
\tand l_receiptdate >= date '{start}'
\tand l_receiptdate < date '{start}' + interval '{window}' {unit}
group by
\tl_shipmode
order by
\tl_shipmode;
limit -1;
""",
    14: """\
select
\t100.00 * sum(case
\t\twhen p_type like 'PROMO%'
\t\t\tthen l_extendedprice * (1 - l_discount)
\t\telse 0
\tend) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
\tlineitem,
\tpart
where
\tl_partkey = p_partkey
\tand l_shipdate >= date '{start}'
\tand l_shipdate < date '{start}' + interval '{window}' {unit};
limit -1;
""",
    15: """\
create view revenue0 (supplier_no, total_revenue) as
\tselect
\t\tl_suppkey,
\t\tsum(l_extendedprice * (1 - l_discount))
\tfrom
\t\tlineitem
\twhere
\t\tl_shipdate >= date '{start}'
\t\tand l_shipdate < date '{start}' + interval '{window}' {unit}
\tgroup by
\t\tl_suppkey;

select
\ts_suppkey,
\ts_name,
\ts_address,
\ts_phone,
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
""",
    18: """\
select
\tc_name,
\tc_custkey,
\to_orderkey,
\to_orderdate,
\to_totalprice,
\tsum(l_quantity)
from
\tcustomer,
\torders,
\tlineitem
where
\to_orderkey in (
\t\tselect
\t\t\tl_orderkey
\t\tfrom
\t\t\tlineitem
\t\tgroup by
\t\t\tl_orderkey having
\t\t\t\tsum(l_quantity) > {quantity}
\t)
\tand c_custkey = o_custkey
\tand o_orderkey = l_orderkey
group by
\tc_name,
\tc_custkey,
\to_orderkey,
\to_orderdate,
\to_totalprice
order by
\to_totalprice desc,
\to_orderdate;
limit 100;
""",
    19: """\
select
\tsum(l_extendedprice* (1 - l_discount)) as revenue
from
\tlineitem,
\tpart
where
\t(
\t\tp_partkey = l_partkey
\t\tand p_brand = '{brand1}'
\t\tand p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
\t\tand l_quantity >= {quantity1} and l_quantity <= {quantity1} + {quantity_band}
\t\tand p_size between 1 and {size1}
\t\tand l_shipmode in ('AIR', 'AIR REG')
\t\tand l_shipinstruct = 'DELIVER IN PERSON'
\t)
\tor
\t(
\t\tp_partkey = l_partkey
\t\tand p_brand = '{brand2}'
\t\tand p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
\t\tand l_quantity >= {quantity2} and l_quantity <= {quantity2} + {quantity_band}
\t\tand p_size between 1 and {size2}
\t\tand l_shipmode in ('AIR', 'AIR REG')
\t\tand l_shipinstruct = 'DELIVER IN PERSON'
\t)
\tor
\t(
\t\tp_partkey = l_partkey
\t\tand p_brand = '{brand3}'
\t\tand p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
\t\tand l_quantity >= {quantity3} and l_quantity <= {quantity3} + {quantity_band}
\t\tand p_size between 1 and {size3}
\t\tand l_shipmode in ('AIR', 'AIR REG')
\t\tand l_shipinstruct = 'DELIVER IN PERSON'
\t);
limit -1;
""",
    20: """\
select
\ts_name,
\ts_address
from
\tsupplier,
\tnation
where
\ts_suppkey in (
\t\tselect
\t\t\tps_suppkey
\t\tfrom
\t\t\tpartsupp
\t\twhere
\t\t\tps_partkey in (
\t\t\t\tselect
\t\t\t\t\tp_partkey
\t\t\t\tfrom
\t\t\t\t\tpart
\t\t\t\twhere
\t\t\t\t\tp_name like '{prefix}%'
\t\t\t)
\t\t\tand ps_availqty > (
\t\t\t\tselect
\t\t\t\t\t0.5 * sum(l_quantity)
\t\t\t\tfrom
\t\t\t\t\tlineitem
\t\t\t\twhere
\t\t\t\t\tl_partkey = ps_partkey
\t\t\t\t\tand l_suppkey = ps_suppkey
\t\t\t\t\tand l_shipdate >= date '{start}'
\t\t\t\t\tand l_shipdate < date '{start}' + interval '{window}' {unit}
\t\t\t)
\t)
\tand s_nationkey = n_nationkey
\tand n_name = '{nation}'
order by
\ts_name;
limit -1;
""",
}
WIDENED_TEMPLATES = sorted(TEMPLATES)

# How each placeholder's value is written into the SQL and read back from it.
# Placeholders not listed are quoted text (market segment, region, nation, ship
# modes, brands and the Q20 name prefix).
KINDS = {"delta": "int", "date": "date", "start": "date", "window": "int", "unit": "unit",
         "discount": "cents", "discount_band": "cents", "quantity": "int", "fraction": "fraction",
         "quantity1": "int", "quantity2": "int", "quantity3": "int", "quantity_band": "int",
         "size1": "int", "size2": "int", "size3": "int"}
PATTERNS = {"int": r"\d+", "date": r"\d{4}-\d{2}-\d{2}", "unit": r"day|month|year", "cents": r"\d\.\d{2}",
            "fraction": r"\d\.\d{10}", "text": r"[^']+"}
PLACEHOLDER = re.compile(r"\{(\w+)\}")
SENTINEL_BASE = 770000
SENTINEL = re.compile(r"7700\d\d")


def kind(name: str) -> str:
    return KINDS.get(name, "text")


def write_value(name: str, value) -> str:
    k = kind(name)
    if k == "cents":
        return f"{value:.2f}"
    if k == "fraction":
        return f"{value:.10f}"  # qgen writes the Q11 fraction with "%11.10f"
    return str(value)


def read_value(name: str, text: str):
    k = kind(name)
    if k == "int":
        return int(text)
    if k in ("cents", "fraction"):
        return float(text)
    return text


def render(template: int, params: dict) -> str:
    """Final SQL of one query: its template with the values filled in, then
    rewritten for PostgreSQL by to_postgres, as every pilot query was."""
    return to_postgres(PLACEHOLDER.sub(lambda m: write_value(m.group(1), params[m.group(1)]),
                                       TEMPLATES[template]))


def matcher(template: int) -> re.Pattern:
    """Regex that matches exactly the final SQL of a query of this template and
    captures its values.

    The template is filled with a distinct run of digits for each placeholder
    and passed through to_postgres, so the regex follows what to_postgres
    really writes (Q1's dropped "(3)", the moved limit, Q15's WITH clause)
    instead of a copy of its rules. A placeholder that occurs twice must hold
    the same value both times."""
    names = list(dict.fromkeys(PLACEHOLDER.findall(TEMPLATES[template])))
    filled = PLACEHOLDER.sub(lambda m: str(SENTINEL_BASE + names.index(m.group(1))), TEMPLATES[template])
    seen: set[str] = set()

    def group(m: re.Match) -> str:
        name = names[int(m.group(0)) - SENTINEL_BASE]
        if name in seen:
            return f"(?P={name})"
        seen.add(name)
        return f"(?P<{name}>{PATTERNS[kind(name)]})"

    return re.compile(SENTINEL.sub(group, re.escape(to_postgres(filled))))


def parse(template: int, sql: str) -> dict:
    """The values of a query of this template, read back from its final SQL."""
    m = matcher(template).fullmatch(sql)
    if m is None:
        raise ValueError(f"not a query of template {template}:\n{sql}")
    return {name: read_value(name, text) for name, text in m.groupdict().items()}


def window_end(params: dict) -> dt.date:
    """First day after a query's date window, start + window in its unit. Adding
    months follows PostgreSQL: a day past the end of the month is clamped."""
    start = dt.date.fromisoformat(params["start"])
    n, unit = params["window"], params["unit"]
    if unit == "day":
        return start + dt.timedelta(days=n)
    year, month = divmod(start.month - 1 + n * (12 if unit == "year" else 1), 12)
    year += start.year
    return dt.date(year, month + 1, min(start.day, calendar.monthrange(year, month + 1)[1]))


def same_query_keys(params: dict, sql: str) -> set:
    """Keys under which two queries of one template count as the same: the
    sorted literal values (dedupe_key), and for a date window the first day
    after the window together with the other values, sorted without their
    names as dedupe_key sorts them. The second key catches a window written in
    other units (3 months and 92 days from the same first day) even when the
    other values come in another order (Q12's two ship modes)."""
    keys: set = {dedupe_key(sql)}
    if "window" in params:
        rest = sorted(str(v) for k, v in params.items() if k not in ("window", "unit", "qgen_seed"))
        keys.add(("window ending", window_end(params).isoformat(), *rest))
    return keys


# ------------------------------------------------------------------ drawing

def log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def uniform_day(rng: random.Random, first: dt.date, last: dt.date) -> dt.date:
    return first + dt.timedelta(days=rng.randint(0, (last - first).days))


def window(rng: random.Random, dates: tuple[dt.date, dt.date]) -> dict:
    """A window of SHORTEST to LONGEST days (log-uniform) that lies inside the
    column's data range."""
    days = round(log_uniform(rng, SHORTEST, LONGEST))
    start = uniform_day(rng, dates[0], dates[1] - dt.timedelta(days=days - 1))
    return {"start": start.isoformat(), "window": days, "unit": "day"}


def draw(template: int, rng: random.Random) -> dict:
    """One widened parameterization of a template (ranges in the module docstring)."""
    if template == 1:
        return {"delta": rng.randint(0, 2400)}
    if template == 3:
        return {"segment": rng.choice(SEGMENTS), "date": uniform_day(rng, *ORDER_DATES).isoformat()}
    if template in (4, 10):
        return window(rng, ORDER_DATES)
    if template == 5:
        return {"region": rng.choice(REGIONS)} | window(rng, ORDER_DATES)
    if template == 6:
        return window(rng, SHIP_DATES) | {"discount": rng.randint(0, 10) / 100,
                                          "discount_band": rng.randint(0, 5) / 100,
                                          "quantity": rng.randint(2, 51)}
    if template == 11:
        return {"nation": rng.choice(NATIONS), "fraction": round(log_uniform(rng, 1e-7, 1e-3), 10)}
    if template == 12:
        first, second = rng.sample(SHIP_MODES, 2)
        return {"shipmode1": first, "shipmode2": second} | window(rng, RECEIPT_DATES)
    if template in (14, 15):
        return window(rng, SHIP_DATES)
    if template == 18:
        return {"quantity": rng.randint(30, 311)}
    if template == 19:
        p = {f"brand{i}": f"Brand#{rng.randint(1, 5)}{rng.randint(1, 5)}" for i in (1, 2, 3)}
        p |= {f"quantity{i}": rng.randint(1, 50) for i in (1, 2, 3)}
        p["quantity_band"] = rng.randint(0, 49)
        return p | {f"size{i}": round(log_uniform(rng, 1, 50)) for i in (1, 2, 3)}
    if template == 20:
        # A prefix that ends at the space is the first word alone.
        name = " ".join(rng.sample(COLORS, 2))
        return ({"prefix": name[: rng.randint(1, len(name))].rstrip()} | window(rng, SHIP_DATES)
                | {"nation": rng.choice(NATIONS)})
    raise ValueError(f"template {template} has no widened parameters")


def widen(template: int, seed: int, taken: set, n: int = PER_TEMPLATE) -> list[dict]:
    """n widened queries of a template, distinct from each other and from every
    query whose same_query_keys are in taken."""
    rng = random.Random(f"widen/{seed}/{template}")
    taken = set(taken)
    out: list[dict] = []
    for _ in range(MAX_TRIES):
        params = draw(template, rng)
        sql = render(template, params)
        keys = same_query_keys(params, sql)
        if keys & taken:
            continue
        taken |= keys
        out.append({"parameters": params, "sql": sql})
        if len(out) == n:
            return out
    raise RuntimeError(f"only {len(out)} distinct widened queries for template {template} "
                       f"after {MAX_TRIES} draws")


def make_workload(seed: int = SEED, std_dir: Path = STD_DIR) -> tuple[list[dict], dict[str, str]]:
    """Manifest entries (sorted by query ID) and the final SQL of every query.

    Standard queries keep their text and ID; for the 13 widened templates their
    values are read back from the text, and the text must be exactly the
    template filled with them."""
    entries: list[dict] = []
    sql: dict[str, str] = {}
    taken: dict[int, set] = defaultdict(set)
    for e in json.loads((std_dir / "manifest.json").read_text(encoding="utf-8")):
        qid, t = e["query_id"], e["template"]
        text = (std_dir / f"{qid}.sql").read_text(encoding="utf-8")
        params = {"qgen_seed": e["qgen_seed"]}
        if t in TEMPLATES:
            values = parse(t, text)
            if render(t, values) != text:
                raise ValueError(f"{qid} is not template {t} filled with its own values")
            params |= values
            taken[t] |= same_query_keys(values, text)
        entries.append({"query_id": qid, "template": t, "variant": "std", "parameters": params})
        sql[qid] = text
    for t in WIDENED_TEMPLATES:
        for i, q in enumerate(widen(t, seed, taken[t]), 1):
            qid = f"q{t:02d}_w{i:02d}"
            entries.append({"query_id": qid, "template": t, "variant": "wide", "parameters": q["parameters"]})
            sql[qid] = q["sql"]
    return sorted(entries, key=lambda e: e["query_id"]), sql


def make_split(entries: list[dict], seed: int = SEED) -> list[dict]:
    """Pool/test assignment of every query: inside every (template, variant)
    group, the query IDs are shuffled with their own seeded stream and the
    first max(1, round(0.3 n)) go to the test set (none for a single query)."""
    groups: dict[tuple[int, str], list[str]] = defaultdict(list)
    for e in entries:
        groups[(e["template"], e["variant"])].append(e["query_id"])
    where: dict[str, str] = {}
    for (t, variant), ids in groups.items():
        ids = sorted(ids)
        random.Random(f"split/{seed}/{t}/{variant}").shuffle(ids)
        n_test = 0 if len(ids) < 2 else max(1, round(TEST_FRAC * len(ids)))
        where |= {q: "test" if i < n_test else "pool" for i, q in enumerate(ids)}
    return [{"query_id": e["query_id"], "template": e["template"], "variant": e["variant"],
             "set": where[e["query_id"]]} for e in sorted(entries, key=lambda e: e["query_id"])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=SEED, help="seed of the widened values and of the split")
    ap.add_argument("--force", action="store_true",
                    help="replace the .sql files already in --out (deleted only once the new workload is built)")
    args = ap.parse_args()

    out = Path(args.out)
    if out.resolve() == STD_DIR.resolve():
        raise SystemExit(f"--out is {STD_DIR}, which holds the pilot queries this workload is built from")
    old = list(out.glob("*.sql"))
    if old and not args.force:
        raise SystemExit(f"{out} already holds {len(old)} .sql files; pass --force to replace them")

    # Build everything before touching --out, so a failure leaves it as it was.
    entries, sql = make_workload(args.seed)
    rows = make_split(entries, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    for f in old:
        f.unlink()
    for e in entries:
        qid = e["query_id"]
        if e["variant"] == "std":
            shutil.copyfile(STD_DIR / f"{qid}.sql", out / f"{qid}.sql")
        else:
            (out / f"{qid}.sql").write_text(sql[qid], encoding="utf-8", newline="\n")
    (out / "manifest.json").write_text(json.dumps(entries, indent=1) + "\n", encoding="utf-8", newline="\n")
    with open(out / "split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "template", "variant", "set"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    count: dict[tuple[int, str, str], int] = defaultdict(int)
    for r in rows:
        count[(r["template"], r["variant"], r["set"])] += 1
    for t in sorted({r["template"] for r in rows}):
        cells = [f"{v} {count[(t, v, 'pool')]}/{count[(t, v, 'test')]}" for v in ("std", "wide")
                 if count[(t, v, "pool")] + count[(t, v, "test")]]
        print(f"template {t:2d}: " + ", ".join(cells) + " (pool/test)")
    print(f"{len(rows)} queries: {sum(r['set'] == 'pool' for r in rows)} pool, "
          f"{sum(r['set'] == 'test' for r in rows)} test")


if __name__ == "__main__":
    main()
