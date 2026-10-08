import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "tpcds_queries", Path(__file__).resolve().parents[1] / "scripts" / "tpcds_queries.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

# Shapes copied from the kit's ansi.tpl and from dsqgen 2.10.0 output with the
# postgres dialect (templates 36, 21, 50 and 39), shortened.
ANSI = """--
-- Legal Notice
--
define __LIMITA = "";
define __LIMITB = "top %d";
define __LIMITC = "";
define _BEGIN = "-- start query " + [_QUERY] + " in stream " + [_STREAM] + " using template " + [_TEMPLATE];
define _END = "-- end query " + [_QUERY] + " in stream " + [_STREAM] + " using template " + [_TEMPLATE];
"""

Q36 = """-- start query 1 in stream 0 using template query36.tpl
select
    sum(ss_net_profit)/sum(ss_ext_sales_price) as gross_margin
   ,i_category
   ,i_class
   ,grouping(i_category)+grouping(i_class) as lochierarchy
   ,rank() over (
 \tpartition by grouping(i_category)+grouping(i_class),
 \tcase when grouping(i_class) = 0 then i_category end
 \torder by sum(ss_net_profit)/sum(ss_ext_sales_price) asc) as rank_within_parent
 from
    store_sales
   ,date_dim       d1
   ,item
   ,store
 where
    d1.d_year = 2002
 and s_state in ('TN','TN','TN','TN',
                 'TN','TN','TN','TN')
 group by rollup(i_category,i_class)
 order by
   lochierarchy desc
  ,case when lochierarchy = 0 then i_category end
  ,rank_within_parent
  limit 100;

-- end query 1 in stream 0 using template query36.tpl
"""

Q21 = """-- start query 1 in stream 0 using template query21.tpl
select  w_warehouse_name
 from inventory, warehouse, item, date_dim
 where i_current_price between 0.99 and 1.49
     and d_date between (cast ('1999-05-26' as date) - 30 days)
                    and (cast ('1999-05-26' as date) + 30 days)
 order by w_warehouse_name
 limit 100;

-- end query 1 in stream 0 using template query21.tpl
"""

Q50 = """-- start query 1 in stream 0 using template query50.tpl
select
   s_store_name
  ,sum(case when (sr_returned_date_sk - ss_sold_date_sk <= 30 ) then 1 else 0 end)  as "30 days"
  ,sum(case when (sr_returned_date_sk - ss_sold_date_sk > 30) and
                 (sr_returned_date_sk - ss_sold_date_sk <= 60) then 1 else 0 end )  as "31-60 days"
from store_sales, store_returns, store
group by s_store_name
limit 100;

-- end query 1 in stream 0 using template query50.tpl
"""

Q39 = """-- start query 1 in stream 0 using template query39.tpl
with inv as
(select w_warehouse_sk, i_item_sk, d_moy, stdev, mean
 from inventory)
select inv1.w_warehouse_sk, inv1.cov
from inv inv1, inv inv2
where inv1.i_item_sk = inv2.i_item_sk
  and inv1.d_moy=3
order by inv1.w_warehouse_sk,inv1.i_item_sk;
with inv as
(select w_warehouse_sk, i_item_sk, d_moy, stdev, mean
 from inventory)
select inv1.w_warehouse_sk, inv1.cov
from inv inv1, inv inv2
where inv1.i_item_sk = inv2.i_item_sk
  and inv1.d_moy=3
  and inv1.cov > 1.5
order by inv1.w_warehouse_sk,inv1.i_item_sk;

-- end query 1 in stream 0 using template query39.tpl
"""

LOG = """BEGIN STREAM 0
Template: query36.tpl
\t__LIMITA.01 =
\t_LIMITC.01 = limit 100
\t_LIMIT.01 = 100
\tSTATE_B.01 = TN
\tSTATE_A.01 = TN
\tSTATENUMBER.01 = 1
\tSTATENUMBER.02 = 3
\tYEAR.01 = 2002
\t_END.01 = -- end query 1 in stream 0 using template query36.tpl
\t_TEMPLATE.01 = query36.tpl

END STREAM 0
"""


def test_dialect_writes_limit_at_the_end():
    d = gen.postgres_dialect(ANSI)
    assert 'define __LIMITA = "";' in d
    assert 'define __LIMITB = "";' in d
    assert 'define __LIMITC = "limit %d";' in d
    assert "top" not in d
    assert 'define _BEGIN = "-- start query "' in d


def test_dialect_needs_every_limit_define():
    with pytest.raises(ValueError):
        gen.postgres_dialect('define __LIMITA = "";\n')


def test_days_become_intervals():
    (sql,), fixes = gen.to_postgres(Q21)
    assert "(cast ('1999-05-26' as date) - interval '30 days')" in sql
    assert "(cast ('1999-05-26' as date) + interval '30 days')" in sql
    assert "30 days)" not in sql
    assert fixes == ["interval"]
    assert sql.count(";") == 1 and sql.endswith("limit 100;\n")
    assert "--" not in sql


def test_quoted_names_are_not_date_arithmetic():
    (sql,), fixes = gen.to_postgres(Q50)
    assert '"30 days"' in sql and '"31-60 days"' in sql
    assert "interval" not in sql
    assert fixes == []


def test_alias_inside_order_by_expression_is_inlined():
    (sql,), fixes = gen.to_postgres(Q36)
    order_by = sql[sql.rindex("order by"):]
    assert "case when (grouping(i_category)+grouping(i_class)) = 0 then i_category end" in order_by
    assert "lochierarchy desc" in order_by          # a bare alias is fine in PostgreSQL
    assert "grouping(i_category)+grouping(i_class) as lochierarchy" in sql
    assert "order by sum(ss_net_profit)/sum(ss_ext_sales_price) asc)" in sql   # window unchanged
    assert fixes == ["order_by_alias"]
    assert sql.rstrip().endswith("limit 100;")


def test_two_statements_become_two_parts():
    statements, fixes = gen.to_postgres(Q39)
    assert len(statements) == 2
    assert all(s.count(";") == 1 and s.endswith(";\n") for s in statements)
    assert "cov > 1.5" not in statements[0] and "cov > 1.5" in statements[1]
    assert fixes == ["split"]


def test_empty_output_is_an_error():
    with pytest.raises(ValueError):
        gen.to_postgres("-- start query 1 in stream 0 using template query1.tpl\n\n")


def test_query_ids_parse_as_template_numbers():
    assert gen.query_ids(7, 1, 1) == ["q07_001"]
    assert gen.query_ids(14, 3, 2) == ["q14_003a", "q14_003b"]
    for template, qid in [(7, "q07_001"), (14, "q14_003a"), (14, "q14_003b"), (99, gen.query_ids(99, 5, 1)[0])]:
        # the expression src/collect.py uses for the template column
        assert int(qid.split("_")[0].lstrip("qQ")) == template


def test_parameter_log_without_internal_variables():
    assert gen.parse_param_log(LOG) == {"STATENUMBER": ["1", "3"], "STATE_A": "TN", "STATE_B": "TN",
                                        "YEAR": "2002"}


def test_duplicates_ignore_list_order_but_not_choices():
    base = "select sum(ss_quantity) from t where s_state in ('TN', 'GA') and d_year = 2000;"
    reordered = "select sum(ss_quantity) from t where s_state in ('GA', 'TN') and d_year = 2000;"
    other_agg = "select avg(ss_quantity) from t where s_state in ('TN', 'GA') and d_year = 2000;"
    other_year = "select sum(ss_quantity) from t where s_state in ('TN', 'GA') and d_year = 2001;"
    assert gen.query_key(base) == gen.query_key(reordered)
    assert gen.query_key(base) != gen.query_key(other_agg)
    assert gen.query_key(base) != gen.query_key(other_year)


def test_version_line_from_help_text():
    usage = "USAGE: qgen2 [options]\n\nNote: When defined in a parameter file (using -p), parmeters should\n"
    header = "qgen2 Population Generator (Version 2.10.0)\nCopyright Transaction Processing Performance Council\n"
    assert gen.version_line(usage + header) == "qgen2 Population Generator (Version 2.10.0)"
    assert gen.version_line(usage) == ""


def test_split_top_level():
    assert gen.split_top_level("a desc, f(b, c), d") == ["a desc", " f(b, c)", " d"]


def two_part_output(year: int, day: int) -> str:
    """dsqgen output of a two-statement template whose part a uses only the year."""
    return (f"-- start query 1 in stream 0 using template query14.tpl\n"
            f"select count(*) from store_sales, date_dim where ss_sold_date_sk = d_date_sk and d_year = {year};\n"
            f"select count(*) from store_sales, date_dim where ss_sold_date_sk = d_date_sk "
            f"and d_year = {year} and d_dom = {day};\n\n"
            f"-- end query 1 in stream 0 using template query14.tpl\n")


def test_a_repeated_part_is_not_written_again():
    # Part a has two possible years, so after two draws only part b is new.
    def run_one(seed):
        return two_part_output(1998 + seed % 2, seed % 28 + 1), {"YEAR": 1998 + seed % 2}

    rng = gen.random.Random(0)
    files, manifest, info = gen.draw_template(14, run_one, 5, 200, rng)
    a = sorted(q for q in files if q.endswith("a"))
    b = sorted(q for q in files if q.endswith("b"))
    assert info["queries"] == 5 and len(b) == 5 and len(a) == 2
    assert info["files_per_part"] == [2, 5] and info["files"] == 7
    assert len({files[q] for q in a}) == 2 and len({files[q] for q in b}) == 5
    assert [e["query_id"] for e in manifest] == sorted(files, key=lambda q: (q[:-1], q[-1]))


def test_a_draw_whose_parts_all_repeat_is_skipped():
    def run_one(seed):
        return two_part_output(1999, 7), {"YEAR": 1999}

    files, _, info = gen.draw_template(14, run_one, 5, 20, gen.random.Random(0))
    assert info["queries"] == 1 and info["tries"] == 20 and sorted(files) == ["q14_001a", "q14_001b"]
