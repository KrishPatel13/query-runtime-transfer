select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1996-10-30'
	and l_shipdate < date '1996-10-30' + interval '35' day
	and l_discount between 0.10 - 0.02 and 0.10 + 0.02
	and l_quantity < 49;
