select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1993-03-07'
	and l_shipdate < date '1993-03-07' + interval '13' day
	and l_discount between 0.04 - 0.03 and 0.04 + 0.03
	and l_quantity < 11;
