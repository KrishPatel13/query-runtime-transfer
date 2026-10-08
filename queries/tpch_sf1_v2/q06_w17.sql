select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1993-11-25'
	and l_shipdate < date '1993-11-25' + interval '88' day
	and l_discount between 0.03 - 0.02 and 0.03 + 0.02
	and l_quantity < 21;
