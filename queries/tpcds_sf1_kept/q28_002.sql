select  *
from (select avg(ss_list_price) B1_LP
            ,count(ss_list_price) B1_CNT
            ,count(distinct ss_list_price) B1_CNTD
      from store_sales
      where ss_quantity between 0 and 5
        and (ss_list_price between 147 and 147+10 
             or ss_coupon_amt between 2206 and 2206+1000
             or ss_wholesale_cost between 33 and 33+20)) B1,
     (select avg(ss_list_price) B2_LP
            ,count(ss_list_price) B2_CNT
            ,count(distinct ss_list_price) B2_CNTD
      from store_sales
      where ss_quantity between 6 and 10
        and (ss_list_price between 109 and 109+10
          or ss_coupon_amt between 14445 and 14445+1000
          or ss_wholesale_cost between 35 and 35+20)) B2,
     (select avg(ss_list_price) B3_LP
            ,count(ss_list_price) B3_CNT
            ,count(distinct ss_list_price) B3_CNTD
      from store_sales
      where ss_quantity between 11 and 15
        and (ss_list_price between 41 and 41+10
          or ss_coupon_amt between 14765 and 14765+1000
          or ss_wholesale_cost between 58 and 58+20)) B3,
     (select avg(ss_list_price) B4_LP
            ,count(ss_list_price) B4_CNT
            ,count(distinct ss_list_price) B4_CNTD
      from store_sales
      where ss_quantity between 16 and 20
        and (ss_list_price between 63 and 63+10
          or ss_coupon_amt between 1072 and 1072+1000
          or ss_wholesale_cost between 10 and 10+20)) B4,
     (select avg(ss_list_price) B5_LP
            ,count(ss_list_price) B5_CNT
            ,count(distinct ss_list_price) B5_CNTD
      from store_sales
      where ss_quantity between 21 and 25
        and (ss_list_price between 65 and 65+10
          or ss_coupon_amt between 16636 and 16636+1000
          or ss_wholesale_cost between 14 and 14+20)) B5,
     (select avg(ss_list_price) B6_LP
            ,count(ss_list_price) B6_CNT
            ,count(distinct ss_list_price) B6_CNTD
      from store_sales
      where ss_quantity between 26 and 30
        and (ss_list_price between 120 and 120+10
          or ss_coupon_amt between 3338 and 3338+1000
          or ss_wholesale_cost between 18 and 18+20)) B6
limit 100;
