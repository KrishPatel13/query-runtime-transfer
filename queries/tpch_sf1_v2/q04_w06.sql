select
	o_orderpriority,
	count(*) as order_count
from
	orders
where
	o_orderdate >= date '1995-12-12'
	and o_orderdate < date '1995-12-12' + interval '32' day
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
