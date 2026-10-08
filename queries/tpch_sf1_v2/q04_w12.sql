select
	o_orderpriority,
	count(*) as order_count
from
	orders
where
	o_orderdate >= date '1994-08-27'
	and o_orderdate < date '1994-08-27' + interval '44' day
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
