select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1993-11-03'
	and l_shipdate < date '1993-11-03' + interval '616' day
	and l_discount between 0.02 - 0.00 and 0.02 + 0.00
	and l_quantity < 18;
