select
	s_name,
	s_address
from
	supplier,
	nation
where
	s_suppkey in (
		select
			ps_suppkey
		from
			partsupp
		where
			ps_partkey in (
				select
					p_partkey
				from
					part
				where
					p_name like 'papaya%'
			)
			and ps_availqty > (
				select
					0.5 * sum(l_quantity)
				from
					lineitem
				where
					l_partkey = ps_partkey
					and l_suppkey = ps_suppkey
					and l_shipdate >= date '1992-04-10'
					and l_shipdate < date '1992-04-10' + interval '86' day
			)
	)
	and s_nationkey = n_nationkey
	and n_name = 'VIETNAM'
order by
	s_name;
