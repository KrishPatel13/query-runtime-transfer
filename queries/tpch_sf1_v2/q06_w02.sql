select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1993-06-17'
	and l_shipdate < date '1993-06-17' + interval '1165' day
	and l_discount between 0.10 - 0.04 and 0.10 + 0.04
	and l_quantity < 44;
