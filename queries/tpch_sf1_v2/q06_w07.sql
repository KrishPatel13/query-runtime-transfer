select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1992-04-21'
	and l_shipdate < date '1992-04-21' + interval '140' day
	and l_discount between 0.04 - 0.02 and 0.04 + 0.02
	and l_quantity < 28;
