# Orders missing values
payment_id 0.004% of column is Nan, could just drop should be negligible
payment 0.004% of column is Nan, could just drop should be negligible
unit_rrp_vat_excl 1.4% of column is Nan
sales_person_id 0.2% of column is Nan
sales_person 0.2% of column is Nan

# Items missing values
Group 3 and Group 4 are 100% Nan, will drop those columns as well as their Group id columns.
Group 2 is 93% Nan, should drop as well
Group 1 is 40% Nan, will keep for now, the categories it does have might be prevalent in the orders table.

understanding: item_name is specific product name, name is brand name
brand_id -> name, same length

Important item columns: ['item_code','item_name','style','name','group0','group1']



