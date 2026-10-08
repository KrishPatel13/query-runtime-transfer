select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1997-07-29'
	and l_shipdate < date '1997-07-29' + interval '20' day
	and l_discount between 0.03 - 0.03 and 0.03 + 0.03
	and l_quantity < 11;
