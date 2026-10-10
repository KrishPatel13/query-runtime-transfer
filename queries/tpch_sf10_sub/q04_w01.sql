select
	o_orderpriority,
	count(*) as order_count
from
	orders
where
	o_orderdate >= date '1994-10-13'
	and o_orderdate < date '1994-10-13' + interval '48' day
	and exists (
		select
			*
		from
			lineitem
		where
			l_orderkey = o_orderkey
			and l_commitdate < l_receiptdate
	)
group by
	o_orderpriority
order by
	o_orderpriority;
