select
	sum(l_extendedprice * l_discount) as revenue
from
	lineitem
where
	l_shipdate >= date '1993-07-15'
	and l_shipdate < date '1993-07-15' + interval '1049' day
	and l_discount between 0.09 - 0.02 and 0.09 + 0.02
	and l_quantity < 17;
