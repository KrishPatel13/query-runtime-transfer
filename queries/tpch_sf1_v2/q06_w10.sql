select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1996-02-01'
	and l_shipdate < date '1996-02-01' + interval '108' day
	and l_discount between 0.06 - 0.04 and 0.06 + 0.04
	and l_quantity < 21;
