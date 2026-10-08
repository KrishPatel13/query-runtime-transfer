-- Indexes added to the TPC-DS database after the load, on top of the primary
-- keys that the kit's tpcds.sql defines and the foreign keys that the kit's
-- tpcds_ri.sql declares (scripts/tpcds_load.sh runs both kit files).
--
-- The physical design follows the TPC-H one in tpch_keys_indexes.sql: keys,
-- plus an index on every foreign-key column and on the date column that
-- queries filter on. Here that means one index for each foreign key in
-- tpcds_ri.sql (95 columns; the 7 foreign keys that are already the first
-- column of their table's primary key are left out, since the primary-key
-- index serves them), plus date_dim (d_date).
--
-- Every index is allowed by the TPC-DS specification (v2.10.0, the version
-- of the kit). An index is an explicit auxiliary data structure (clause
-- 2.5.1.3), and clause 2.5.3.2 lets one hold data from a single base table
-- only: primary-key columns, row pointers, and at most one of a foreign key,
-- a column of date type, or a business key. Each index below holds one
-- foreign-key column or one date column of one table, plus PostgreSQL's row
-- pointers. Clause 2.5.3.5 forbids structures over dimension tables only
-- unless 2.5.3.2 allows them, which it does for these. Clause 2.5.4.1 allows
-- primary-key, foreign-key and NOT NULL constraints. No other index is added,
-- although clause 2.5.3.3 would allow more on catalog_sales and
-- catalog_returns.

CREATE INDEX call_center_cc_closed_date_sk_idx ON call_center (cc_closed_date_sk);
CREATE INDEX call_center_cc_open_date_sk_idx ON call_center (cc_open_date_sk);
CREATE INDEX catalog_page_cp_end_date_sk_idx ON catalog_page (cp_end_date_sk);
CREATE INDEX catalog_page_cp_start_date_sk_idx ON catalog_page (cp_start_date_sk);
CREATE INDEX catalog_returns_cr_call_center_sk_idx ON catalog_returns (cr_call_center_sk);
CREATE INDEX catalog_returns_cr_catalog_page_sk_idx ON catalog_returns (cr_catalog_page_sk);
CREATE INDEX catalog_returns_cr_reason_sk_idx ON catalog_returns (cr_reason_sk);
CREATE INDEX catalog_returns_cr_refunded_addr_sk_idx ON catalog_returns (cr_refunded_addr_sk);
CREATE INDEX catalog_returns_cr_refunded_cdemo_sk_idx ON catalog_returns (cr_refunded_cdemo_sk);
CREATE INDEX catalog_returns_cr_refunded_customer_sk_idx ON catalog_returns (cr_refunded_customer_sk);
CREATE INDEX catalog_returns_cr_refunded_hdemo_sk_idx ON catalog_returns (cr_refunded_hdemo_sk);
CREATE INDEX catalog_returns_cr_returned_date_sk_idx ON catalog_returns (cr_returned_date_sk);
CREATE INDEX catalog_returns_cr_returned_time_sk_idx ON catalog_returns (cr_returned_time_sk);
CREATE INDEX catalog_returns_cr_returning_addr_sk_idx ON catalog_returns (cr_returning_addr_sk);
CREATE INDEX catalog_returns_cr_returning_cdemo_sk_idx ON catalog_returns (cr_returning_cdemo_sk);
CREATE INDEX catalog_returns_cr_returning_customer_sk_idx ON catalog_returns (cr_returning_customer_sk);
CREATE INDEX catalog_returns_cr_returning_hdemo_sk_idx ON catalog_returns (cr_returning_hdemo_sk);
CREATE INDEX catalog_returns_cr_ship_mode_sk_idx ON catalog_returns (cr_ship_mode_sk);
CREATE INDEX catalog_returns_cr_warehouse_sk_idx ON catalog_returns (cr_warehouse_sk);
CREATE INDEX catalog_sales_cs_bill_addr_sk_idx ON catalog_sales (cs_bill_addr_sk);
CREATE INDEX catalog_sales_cs_bill_cdemo_sk_idx ON catalog_sales (cs_bill_cdemo_sk);
CREATE INDEX catalog_sales_cs_bill_customer_sk_idx ON catalog_sales (cs_bill_customer_sk);
CREATE INDEX catalog_sales_cs_bill_hdemo_sk_idx ON catalog_sales (cs_bill_hdemo_sk);
CREATE INDEX catalog_sales_cs_call_center_sk_idx ON catalog_sales (cs_call_center_sk);
CREATE INDEX catalog_sales_cs_catalog_page_sk_idx ON catalog_sales (cs_catalog_page_sk);
CREATE INDEX catalog_sales_cs_promo_sk_idx ON catalog_sales (cs_promo_sk);
CREATE INDEX catalog_sales_cs_ship_addr_sk_idx ON catalog_sales (cs_ship_addr_sk);
CREATE INDEX catalog_sales_cs_ship_cdemo_sk_idx ON catalog_sales (cs_ship_cdemo_sk);
CREATE INDEX catalog_sales_cs_ship_customer_sk_idx ON catalog_sales (cs_ship_customer_sk);
CREATE INDEX catalog_sales_cs_ship_date_sk_idx ON catalog_sales (cs_ship_date_sk);
CREATE INDEX catalog_sales_cs_ship_hdemo_sk_idx ON catalog_sales (cs_ship_hdemo_sk);
CREATE INDEX catalog_sales_cs_ship_mode_sk_idx ON catalog_sales (cs_ship_mode_sk);
CREATE INDEX catalog_sales_cs_sold_date_sk_idx ON catalog_sales (cs_sold_date_sk);
CREATE INDEX catalog_sales_cs_sold_time_sk_idx ON catalog_sales (cs_sold_time_sk);
CREATE INDEX catalog_sales_cs_warehouse_sk_idx ON catalog_sales (cs_warehouse_sk);
CREATE INDEX customer_c_current_addr_sk_idx ON customer (c_current_addr_sk);
CREATE INDEX customer_c_current_cdemo_sk_idx ON customer (c_current_cdemo_sk);
CREATE INDEX customer_c_current_hdemo_sk_idx ON customer (c_current_hdemo_sk);
CREATE INDEX customer_c_first_sales_date_sk_idx ON customer (c_first_sales_date_sk);
CREATE INDEX customer_c_first_shipto_date_sk_idx ON customer (c_first_shipto_date_sk);
CREATE INDEX household_demographics_hd_income_band_sk_idx ON household_demographics (hd_income_band_sk);
CREATE INDEX inventory_inv_item_sk_idx ON inventory (inv_item_sk);
CREATE INDEX inventory_inv_warehouse_sk_idx ON inventory (inv_warehouse_sk);
CREATE INDEX promotion_p_end_date_sk_idx ON promotion (p_end_date_sk);
CREATE INDEX promotion_p_item_sk_idx ON promotion (p_item_sk);
CREATE INDEX promotion_p_start_date_sk_idx ON promotion (p_start_date_sk);
CREATE INDEX store_s_closed_date_sk_idx ON store (s_closed_date_sk);
CREATE INDEX store_returns_sr_addr_sk_idx ON store_returns (sr_addr_sk);
CREATE INDEX store_returns_sr_cdemo_sk_idx ON store_returns (sr_cdemo_sk);
CREATE INDEX store_returns_sr_customer_sk_idx ON store_returns (sr_customer_sk);
CREATE INDEX store_returns_sr_hdemo_sk_idx ON store_returns (sr_hdemo_sk);
CREATE INDEX store_returns_sr_reason_sk_idx ON store_returns (sr_reason_sk);
CREATE INDEX store_returns_sr_return_time_sk_idx ON store_returns (sr_return_time_sk);
CREATE INDEX store_returns_sr_returned_date_sk_idx ON store_returns (sr_returned_date_sk);
CREATE INDEX store_returns_sr_store_sk_idx ON store_returns (sr_store_sk);
CREATE INDEX store_sales_ss_addr_sk_idx ON store_sales (ss_addr_sk);
CREATE INDEX store_sales_ss_cdemo_sk_idx ON store_sales (ss_cdemo_sk);
CREATE INDEX store_sales_ss_customer_sk_idx ON store_sales (ss_customer_sk);
CREATE INDEX store_sales_ss_hdemo_sk_idx ON store_sales (ss_hdemo_sk);
CREATE INDEX store_sales_ss_promo_sk_idx ON store_sales (ss_promo_sk);
CREATE INDEX store_sales_ss_sold_date_sk_idx ON store_sales (ss_sold_date_sk);
CREATE INDEX store_sales_ss_sold_time_sk_idx ON store_sales (ss_sold_time_sk);
CREATE INDEX store_sales_ss_store_sk_idx ON store_sales (ss_store_sk);
CREATE INDEX web_page_wp_access_date_sk_idx ON web_page (wp_access_date_sk);
CREATE INDEX web_page_wp_creation_date_sk_idx ON web_page (wp_creation_date_sk);
CREATE INDEX web_returns_wr_reason_sk_idx ON web_returns (wr_reason_sk);
CREATE INDEX web_returns_wr_refunded_addr_sk_idx ON web_returns (wr_refunded_addr_sk);
CREATE INDEX web_returns_wr_refunded_cdemo_sk_idx ON web_returns (wr_refunded_cdemo_sk);
CREATE INDEX web_returns_wr_refunded_customer_sk_idx ON web_returns (wr_refunded_customer_sk);
CREATE INDEX web_returns_wr_refunded_hdemo_sk_idx ON web_returns (wr_refunded_hdemo_sk);
CREATE INDEX web_returns_wr_returned_date_sk_idx ON web_returns (wr_returned_date_sk);
CREATE INDEX web_returns_wr_returned_time_sk_idx ON web_returns (wr_returned_time_sk);
CREATE INDEX web_returns_wr_returning_addr_sk_idx ON web_returns (wr_returning_addr_sk);
CREATE INDEX web_returns_wr_returning_cdemo_sk_idx ON web_returns (wr_returning_cdemo_sk);
CREATE INDEX web_returns_wr_returning_customer_sk_idx ON web_returns (wr_returning_customer_sk);
CREATE INDEX web_returns_wr_returning_hdemo_sk_idx ON web_returns (wr_returning_hdemo_sk);
CREATE INDEX web_returns_wr_web_page_sk_idx ON web_returns (wr_web_page_sk);
CREATE INDEX web_sales_ws_bill_addr_sk_idx ON web_sales (ws_bill_addr_sk);
CREATE INDEX web_sales_ws_bill_cdemo_sk_idx ON web_sales (ws_bill_cdemo_sk);
CREATE INDEX web_sales_ws_bill_customer_sk_idx ON web_sales (ws_bill_customer_sk);
CREATE INDEX web_sales_ws_bill_hdemo_sk_idx ON web_sales (ws_bill_hdemo_sk);
CREATE INDEX web_sales_ws_promo_sk_idx ON web_sales (ws_promo_sk);
CREATE INDEX web_sales_ws_ship_addr_sk_idx ON web_sales (ws_ship_addr_sk);
CREATE INDEX web_sales_ws_ship_cdemo_sk_idx ON web_sales (ws_ship_cdemo_sk);
CREATE INDEX web_sales_ws_ship_customer_sk_idx ON web_sales (ws_ship_customer_sk);
CREATE INDEX web_sales_ws_ship_date_sk_idx ON web_sales (ws_ship_date_sk);
CREATE INDEX web_sales_ws_ship_hdemo_sk_idx ON web_sales (ws_ship_hdemo_sk);
CREATE INDEX web_sales_ws_ship_mode_sk_idx ON web_sales (ws_ship_mode_sk);
CREATE INDEX web_sales_ws_sold_date_sk_idx ON web_sales (ws_sold_date_sk);
CREATE INDEX web_sales_ws_sold_time_sk_idx ON web_sales (ws_sold_time_sk);
CREATE INDEX web_sales_ws_warehouse_sk_idx ON web_sales (ws_warehouse_sk);
CREATE INDEX web_sales_ws_web_page_sk_idx ON web_sales (ws_web_page_sk);
CREATE INDEX web_sales_ws_web_site_sk_idx ON web_sales (ws_web_site_sk);
CREATE INDEX web_site_web_close_date_sk_idx ON web_site (web_close_date_sk);
CREATE INDEX web_site_web_open_date_sk_idx ON web_site (web_open_date_sk);

CREATE INDEX date_dim_d_date_idx ON date_dim (d_date);
