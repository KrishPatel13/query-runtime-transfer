select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1994-11-10'
	and l_shipdate < date '1994-11-10' + interval '442' day
	and l_discount between 0.06 - 0.02 and 0.06 + 0.02
	and l_quantity < 38;
