select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1992-03-13'
	and l_shipdate < date '1992-03-13' + interval '286' day
	and l_discount between 0.08 - 0.04 and 0.08 + 0.04
	and l_quantity < 11;
