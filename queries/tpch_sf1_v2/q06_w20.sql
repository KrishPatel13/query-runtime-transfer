select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1993-02-21'
	and l_shipdate < date '1993-02-21' + interval '386' day
	and l_discount between 0.00 - 0.00 and 0.00 + 0.00
	and l_quantity < 23;
