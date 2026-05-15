# Databricks notebook source
# MAGIC %md
# MAGIC # Synthetic Retail Sales Dataset
# MAGIC
# MAGIC Builds a star-schema retail dataset (4 dimensions + 1 fact) using **Spark-native APIs only** —
# MAGIC no Python loops, no `collect()`, no driver-side row construction. Everything scales horizontally.
# MAGIC
# MAGIC **Pattern used throughout:**
# MAGIC - `spark.range(N)` generates the row skeleton in parallel.
# MAGIC - `hash(col, "salt") % K` gives deterministic, repeatable categorical assignment (same input → same output across runs).
# MAGIC - `rand(seed=...)` gives reproducible numeric noise.
# MAGIC - Foreign keys in the fact table are derived via `hash(transaction_id, salt) % dim_size + 1`,
# MAGIC   which guarantees every FK lands inside a valid dimension key range — so joins never drop rows.

# COMMAND ----------

# DBTITLE 1,Cell 2
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("synthetic_sales").getOrCreate()

# Row counts — tune these for your cluster size.
NUM_STORES = 200
NUM_PRODUCTS = 5_000
NUM_CUSTOMERS = 100_000
NUM_PROMOTIONS = 50
NUM_TRANSACTIONS = 1_000_000

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. `dim_store`
# MAGIC
# MAGIC One row per store. `country`, `region`, and `store_type` are picked from fixed arrays using
# MAGIC `element_at(array, idx)` where `idx = pmod(hash(...), len) + 1`.
# MAGIC
# MAGIC We salt each `hash()` call with a different literal (`"region"`, `"type"`) so the three
# MAGIC categorical columns vary independently — otherwise every store with the same `store_id` hash
# MAGIC bucket would land in the same country *and* region *and* type, which isn't realistic.

# COMMAND ----------

# DBTITLE 1,Cell 4
dim_store = (
    spark.range(1, NUM_STORES + 1)
    .withColumnRenamed("id", "store_id")
    .withColumn(
        "country",
        F.element_at(
            F.array(F.lit("US"), F.lit("CA"), F.lit("MX"), F.lit("UK"), F.lit("DE"), F.lit("FR"), F.lit("JP")),
            (F.pmod(F.hash("store_id"), F.lit(7)) + F.lit(1)).cast("int"),
        ),
    )
    .withColumn(
        "region",
        F.element_at(
            F.array(F.lit("North"), F.lit("South"), F.lit("East"), F.lit("West"), F.lit("Central")),
            (F.pmod(F.hash(F.col("store_id"), F.lit("region")), F.lit(5)) + F.lit(1)).cast("int"),
        ),
    )
    .withColumn(
        "store_type",
        F.element_at(
            F.array(F.lit("flagship"), F.lit("standard"), F.lit("outlet"), F.lit("popup")),
            (F.pmod(F.hash(F.col("store_id"), F.lit("type")), F.lit(4)) + F.lit(1)).cast("int"),
        ),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. `dim_product`
# MAGIC
# MAGIC `unit_cost` is sampled uniformly in `[1.0, 191.0]` via `rand() * 190 + 1`. The seed makes the
# MAGIC dataset reproducible — re-running the notebook gives identical costs.

# COMMAND ----------

dim_product = (
    spark.range(1, NUM_PRODUCTS + 1)
    .withColumnRenamed("id", "product_id")
    .withColumn(
        "category",
        F.element_at(
            F.array(
                F.lit("electronics"), F.lit("apparel"), F.lit("grocery"), F.lit("home"),
                F.lit("toys"), F.lit("beauty"), F.lit("sports"), F.lit("books"),
            ),
            (F.pmod(F.hash("product_id"), F.lit(8)) + F.lit(1)).cast("int"),
        ),
    )
    .withColumn(
        "brand",
        F.concat(
            F.lit("brand_"),
            F.pmod(F.hash(F.col("product_id"), F.lit("brand")), F.lit(100)).cast("string"),
        ),
    )
    .withColumn("unit_cost", F.round((F.rand(seed=11) * F.lit(190.0) + F.lit(1.0)).cast("double"), 2))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `dim_customer`
# MAGIC
# MAGIC `signup_date` is built by adding a deterministic day offset (0–2919, ~8 years) to a base date.
# MAGIC Using `pmod(hash(...), 2920)` keeps the date stable per `customer_id` — useful for joins and
# MAGIC for testing slowly-changing-dimension logic later.

# COMMAND ----------

dim_customer = (
    spark.range(1, NUM_CUSTOMERS + 1)
    .withColumnRenamed("id", "customer_id")
    .withColumn(
        "loyalty_tier",
        F.element_at(
            F.array(F.lit("bronze"), F.lit("silver"), F.lit("gold"), F.lit("platinum")),
            (F.pmod(F.hash("customer_id"), F.lit(4)) + F.lit(1)).cast("int"),
        ),
    )
    .withColumn(
        "signup_date",
        F.expr("date_add(to_date('2018-01-01'), cast(pmod(hash(customer_id, 'signup'), 2920) as int))"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. `dim_promotion`
# MAGIC
# MAGIC `discount_pct` is in `[0.0, 0.5]` — nothing offers more than a 50% discount.

# COMMAND ----------

dim_promotion = (
    spark.range(1, NUM_PROMOTIONS + 1)
    .withColumnRenamed("id", "promotion_id")
    .withColumn(
        "promotion_type",
        F.element_at(
            F.array(F.lit("bogo"), F.lit("percent_off"), F.lit("flat_off"), F.lit("clearance"), F.lit("loyalty")),
            (F.pmod(F.hash("promotion_id"), F.lit(5)) + F.lit(1)).cast("int"),
        ),
    )
    .withColumn("discount_pct", F.round((F.rand(seed=22) * F.lit(0.5)).cast("double"), 3))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. `fact_sales`
# MAGIC
# MAGIC Generates `NUM_TRANSACTIONS` rows. Key design choices:
# MAGIC
# MAGIC - **Foreign keys** use `pmod(hash(transaction_id, salt), N) + 1`. This produces an integer in
# MAGIC   `[1, N]` — guaranteed to match a dimension PK. Different salts per FK ensure the keys are
# MAGIC   independent (a store's hash bucket doesn't dictate the product's hash bucket).
# MAGIC - **`promotion_id`** is `NULL` ~70% of the time. Most real transactions aren't promoted, and
# MAGIC   we want to test that downstream revenue math handles `NULL` discounts correctly.
# MAGIC - **`transaction_ts`** is a deterministic timestamp anywhere in calendar year 2024. We add a
# MAGIC   number of seconds (0 to 31,536,000 = 365 days) to a fixed midnight base.
# MAGIC - **`transaction_date`** is derived from the timestamp and used as the partition column.

# COMMAND ----------

fact_sales = (
    spark.range(1, NUM_TRANSACTIONS + 1)
    .withColumnRenamed("id", "transaction_id")
    .withColumn(
        "store_id",
        F.pmod(F.hash(F.col("transaction_id"), F.lit("s")), F.lit(NUM_STORES)) + F.lit(1),
    )
    .withColumn(
        "product_id",
        F.pmod(F.hash(F.col("transaction_id"), F.lit("p")), F.lit(NUM_PRODUCTS)) + F.lit(1),
    )
    .withColumn(
        "customer_id",
        F.pmod(F.hash(F.col("transaction_id"), F.lit("c")), F.lit(NUM_CUSTOMERS)) + F.lit(1),
    )
    .withColumn(
        "promotion_id",
        F.when(
            F.rand(seed=33) < F.lit(0.3),
            F.pmod(F.hash(F.col("transaction_id"), F.lit("pr")), F.lit(NUM_PROMOTIONS)) + F.lit(1),
        ).otherwise(F.lit(None).cast("long")),
    )
    .withColumn("quantity", (F.floor(F.rand(seed=44) * F.lit(10)) + F.lit(1)).cast("int"))
    .withColumn("unit_price", F.round((F.rand(seed=55) * F.lit(490.0) + F.lit(10.0)).cast("double"), 2))
    .withColumn(
        "transaction_ts",
        F.expr(
            "to_timestamp('2024-01-01 00:00:00') + "
            "make_interval(0, 0, 0, 0, 0, 0, cast(pmod(hash(transaction_id, 'ts'), 31536000) as double))"
        ),
    )
    .withColumn("transaction_date", F.to_date("transaction_ts"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Enrichment: join fact to dimensions, compute revenue & cost
# MAGIC
# MAGIC All joins are `LEFT` so we keep every fact row even if a key is missing (only `promotion_id`
# MAGIC can legitimately be `NULL` here, but using LEFT is defensive).
# MAGIC
# MAGIC - **`revenue` = `quantity * unit_price * (1 - discount_pct)`**, with `coalesce(discount_pct, 0)`
# MAGIC   so unpromoted transactions get full price.
# MAGIC - **`cost` = `quantity * unit_cost`** — `unit_cost` comes from `dim_product`.

# COMMAND ----------

enriched_sales = (
    fact_sales.alias("f")
    .join(dim_store.alias("s"), "store_id", "left")
    .join(dim_product.alias("p"), "product_id", "left")
    .join(dim_customer.alias("c"), "customer_id", "left")
    .join(dim_promotion.alias("pr"), "promotion_id", "left")
    .withColumn(
        "revenue",
        F.round(
            F.col("quantity")
            * F.col("unit_price")
            * (F.lit(1.0) - F.coalesce(F.col("discount_pct"), F.lit(0.0))),
            2,
        ),
    )
    .withColumn("cost", F.round(F.col("quantity") * F.col("unit_cost"), 2))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Persist as managed tables
# MAGIC
# MAGIC Fact tables are partitioned by `transaction_date` for efficient date-range pruning.

# COMMAND ----------

dim_store.write.mode("overwrite").saveAsTable("dim_store")
dim_product.write.mode("overwrite").saveAsTable("dim_product")
dim_customer.write.mode("overwrite").saveAsTable("dim_customer")
dim_promotion.write.mode("overwrite").saveAsTable("dim_promotion")
fact_sales.write.mode("overwrite").partitionBy("transaction_date").saveAsTable("fact_sales")
enriched_sales.write.mode("overwrite").partitionBy("transaction_date").saveAsTable("fact_sales_enriched")

# COMMAND ----------

display(spark.table("dim_store"))
display(spark.table("dim_product"))
display(spark.table("dim_customer"))
display(spark.table("dim_promotion"))
display(spark.table("fact_sales"))
display(spark.table("fact_sales_enriched"))
