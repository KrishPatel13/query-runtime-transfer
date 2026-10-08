select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1995-10-05'
	and l_shipdate < date '1995-10-05' + interval '11' day
	and l_discount between 0.09 - 0.03 and 0.09 + 0.03
	and l_quantity < 19;
