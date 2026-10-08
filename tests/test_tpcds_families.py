import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "tpcds_families", Path(__file__).resolve().parents[1] / "scripts" / "tpcds_families.py")
fam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fam)

STORE = """select i_item_id, avg(ss_quantity) agg1
from store_sales, date_dim, item
where ss_sold_date_sk = d_date_sk and ss_item_sk = i_item_sk and d_year = {y}
  and i_category in ('Books', 'Home') and ss_list_price between {lo} and {hi}
group by i_item_id order by i_item_id limit 100;
"""
CATALOG = STORE.replace("ss_", "cs_").replace("store_sales", "catalog_sales")
OTHER = """select c_customer_id, count(*) from customer, customer_address
where c_current_addr_sk = ca_address_sk and ca_state = '{s}' group by c_customer_id having count(*) > {n};
"""


def test_literals_are_masked():
    assert fam.tokens("select 'TN', 12, 3.5 from t where a <= .5") == \
        ["select", "?", ",", "?", ",", "?", "from", "t", "where", "a", "<=", "?"]


def test_near_copies_share_a_family_and_others_do_not():
    texts = {37: STORE.format(y=2000, lo=10, hi=40), 82: CATALOG.format(y=1999, lo=62, hi=92),
             5: OTHER.format(s="TN", n=3)}
    families, pairs = fam.families(texts)
    assert families == {5: 5, 37: 37, 82: 37}
    assert pairs[0]["a"] == 37 and pairs[0]["b"] == 82 and pairs[0]["similarity"] >= fam.THRESHOLD


def test_annotate_writes_the_family_of_every_query(tmp_path):
    manifest = []
    for t, text in ((37, STORE), (82, CATALOG)):
        for n in (1, 2):
            qid = f"q{t:02d}_{n:03d}"
            (tmp_path / f"{qid}.sql").write_text(text.format(y=1998 + n, lo=n, hi=n + 30), encoding="utf-8")
            manifest.append({"query_id": qid, "template": t, "part": None})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rec = fam.annotate(tmp_path)
    out = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert {e["family"] for e in out} == {37} and rec["families"] == 1
    assert json.loads((tmp_path / "families.json").read_text(encoding="utf-8"))["multi_template_families"] == \
        {"37": [37, 82]}
