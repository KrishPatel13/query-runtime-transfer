select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1993-11-01'
	and l_shipdate < date '1993-11-01' + interval '451' day
	and l_discount between 0.02 - 0.05 and 0.02 + 0.05
	and l_quantity < 8;
