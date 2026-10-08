select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1994-04-27'
	and l_shipdate < date '1994-04-27' + interval '1009' day
	and l_discount between 0.04 - 0.05 and 0.04 + 0.05
	and l_quantity < 14;
