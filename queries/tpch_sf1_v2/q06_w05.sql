select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1992-10-03'
	and l_shipdate < date '1992-10-03' + interval '259' day
	and l_discount between 0.04 - 0.03 and 0.04 + 0.03
	and l_quantity < 8;
