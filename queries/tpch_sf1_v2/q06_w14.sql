select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1992-12-18'
	and l_shipdate < date '1992-12-18' + interval '13' day
	and l_discount between 0.06 - 0.03 and 0.06 + 0.03
	and l_quantity < 2;
