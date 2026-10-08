select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1995-05-18'
	and l_shipdate < date '1995-05-18' + interval '45' day
	and l_discount between 0.01 - 0.04 and 0.01 + 0.04
	and l_quantity < 13;
