# Databricks notebook source
# MAGIC %md
# MAGIC # Synthetic Retail Sales Dataset — 100M-row scale build
# MAGIC
# MAGIC Builds a star-schema retail dataset (4 dimensions + 1 fact) using **Spark-native APIs only** —
# MAGIC no Python loops, no `collect()`, no driver-side row construction. Everything scales horizontally.
# MAGIC
# MAGIC **Pattern used throughout:**
# MAGIC - `spark.range(N, numPartitions=...)` generates the row skeleton in parallel with an explicit
# MAGIC   partition count so we don't rely on cluster defaults — important at 100M-row scale.
# MAGIC - `hash(col, "salt") % K` gives deterministic, repeatable categorical assignment.
# MAGIC - `rand(seed=...)` gives reproducible numeric noise.
# MAGIC - Foreign keys in the fact table are derived via `hash(transaction_id, salt) % dim_size + 1`,
# MAGIC   guaranteeing every FK lands inside a valid dimension key range — joins never drop rows.
# MAGIC
# MAGIC **Performance theme:** at 100M rows the bottlenecks shift from *generation* (CPU-bound, parallelizable)
# MAGIC to *shuffle* (network-bound) and *write fan-out* (one task writing many partition dirs = small files).
# MAGIC The code below is engineered around those two costs.

# COMMAND ----------

# DBTITLE 1,Imports and scale configuration
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast

spark = SparkSession.builder.appName("synthetic_sales").getOrCreate()

# Row counts.
# Dimensions are intentionally tiny so they fit comfortably under the broadcast-join threshold
# (default spark.sql.autoBroadcastJoinThreshold = 10MB). All four dim tables together are <1MB serialized,
# so every fact->dim join becomes a map-side broadcast join — zero shuffle.
NUM_STORES = 200
NUM_PRODUCTS = 5_000
NUM_CUSTOMERS = 100_000
NUM_PROMOTIONS = 50

# Fact table at 100M rows.
NUM_TRANSACTIONS = 100_000_000

# Explicit partition count for fact generation.
# Rationale: target ~500K rows per partition. At ~10 columns and mostly numeric/short-string data,
# this lands each task around 50–100MB pre-write, which is the sweet spot for Spark task scheduling
# (small enough to avoid GC pressure, large enough to amortize task-launch overhead).
# 200 partitions also matches the default spark.sql.shuffle.partitions, so any downstream shuffle
# stays balanced without us tuning shuffle.partitions separately.
FACT_NUM_PARTITIONS = 200

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. `dim_store`
# MAGIC
# MAGIC One row per store. `country`, `region`, and `store_type` are picked from fixed arrays using
# MAGIC `element_at(array, idx)` where `idx = pmod(hash(...), len) + 1`.
# MAGIC
# MAGIC We salt each `hash()` call with a different literal (`"region"`, `"type"`) so the three
# MAGIC categorical columns vary independently.
# MAGIC
# MAGIC **Perf:** dimensions stay on a single partition — they're tiny and we want them broadcastable
# MAGIC without an extra coalesce step at write time.

# COMMAND ----------

# DBTITLE 1,dim_store
# numPartitions=1 — 200 rows. More partitions would just create empty tasks and metadata overhead.
dim_store = (
    spark.range(1, NUM_STORES + 1, step=1, numPartitions=1)
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
# MAGIC `unit_cost` is sampled uniformly in `[1.0, 191.0]` via `rand() * 190 + 1`. Seeded for reproducibility.
# MAGIC
# MAGIC **Perf:** 5K rows still fits trivially on one partition (<1MB), and keeping it that way
# MAGIC avoids the small-files problem when the table is later read for broadcasting.

# COMMAND ----------

dim_product = (
    spark.range(1, NUM_PRODUCTS + 1, step=1, numPartitions=1)
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
# MAGIC
# MAGIC **Perf:** 100K rows is still well under the broadcast threshold (~3MB). We use 2 partitions
# MAGIC purely to parallelize generation across two cores — the table coalesces back during the
# MAGIC broadcast collect, so partition count here doesn't affect the join plan.

# COMMAND ----------

dim_customer = (
    spark.range(1, NUM_CUSTOMERS + 1, step=1, numPartitions=2)
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

# Single partition: 50 rows.
dim_promotion = (
    spark.range(1, NUM_PROMOTIONS + 1, step=1, numPartitions=1)
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
# MAGIC ## 5. `fact_sales` — 100M rows
# MAGIC
# MAGIC Key design choices:
# MAGIC
# MAGIC - **Foreign keys** use `pmod(hash(transaction_id, salt), N) + 1`. Different salts per FK ensure
# MAGIC   keys are independent across dimensions.
# MAGIC - **`promotion_id`** is `NULL` ~70% of the time (most real transactions aren't promoted).
# MAGIC - **`transaction_ts`** is a deterministic timestamp anywhere in calendar year 2024.
# MAGIC - **`transaction_date`** is the partition column for the Delta write.
# MAGIC
# MAGIC **Perf at 100M scale:**
# MAGIC - `numPartitions=FACT_NUM_PARTITIONS` (200) gives us ~500K rows per task. Without an explicit
# MAGIC   value, `spark.range` uses `spark.default.parallelism`, which is cluster-dependent and often
# MAGIC   too low (under-utilizing cores) or too high (small-task overhead) for this dataset size.
# MAGIC - All columns are pure expressions over `transaction_id` plus a few seeded `rand()` calls —
# MAGIC   no shuffle, no UDFs. The whole stage is narrow and pipelined.
# MAGIC - We do NOT cache the fact DataFrame: it's a single linear pipeline that's read exactly twice
# MAGIC   (once for the unenriched `fact_sales` write, once for the enriched join). Caching 100M rows
# MAGIC   would cost more memory than the recomputation costs CPU.

# COMMAND ----------

fact_sales = (
    spark.range(1, NUM_TRANSACTIONS + 1, step=1, numPartitions=FACT_NUM_PARTITIONS)
    .withColumnRenamed("id", "transaction_id")
    # Foreign keys: deterministic, uniformly distributed across each dim's PK range.
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
# MAGIC ## 6. Enrichment: project, then join, then compute
# MAGIC
# MAGIC **Perf rules applied here:**
# MAGIC
# MAGIC 1. **Project before join.** We `select(...)` only the columns each dim contributes to the
# MAGIC    enrichment. `dim_store` carries 4 columns total but the fact only needs `country`, `region`,
# MAGIC    `store_type` — dropping `store_id` from the projection keeps the join key but trims the
# MAGIC    broadcast payload. Same idea for the other dims. Smaller broadcasts = less serialization,
# MAGIC    less network, less executor memory pressure.
# MAGIC 2. **Explicit `broadcast(...)` hints.** Even though dims are well under the auto-broadcast
# MAGIC    threshold, the hint is defensive: if a future change pushes a dim past the threshold, the
# MAGIC    plan would silently degrade to a sort-merge join (full shuffle of the 100M-row fact). The
# MAGIC    hint forces a broadcast hash join and surfaces the problem as an OOM instead of silent
# MAGIC    slowdown.
# MAGIC 3. **`LEFT` joins.** Defensive — keep every fact row even if a key is unexpectedly missing.
# MAGIC    Only `promotion_id` is legitimately nullable.
# MAGIC 4. **All numeric columns derived in one pass.** `gross_revenue`, `discount_amount`,
# MAGIC    `net_revenue`, `cost`, `gross_margin` are all computed in a single projection. Spark's
# MAGIC    Catalyst optimizer collapses chained `withColumn` calls into one project node anyway,
# MAGIC    but keeping them in one block makes the dependency chain explicit to readers.

# COMMAND ----------

# Project each dim down to (join_key + only the columns enrichment needs).
# This shrinks the broadcast hash table on every executor — at 100M-row scale, broadcast payload size
# directly impacts executor heap pressure during the join.
dim_store_min = dim_store.select("store_id", "country", "region", "store_type")
dim_product_min = dim_product.select("product_id", "category", "brand", "unit_cost")
# dim_customer's "signup_date" isn't needed downstream by the metrics notebook, so drop it from the
# broadcast. We keep "loyalty_tier" since it's a common slicing dimension for retail metrics.
dim_customer_min = dim_customer.select("customer_id", "loyalty_tier")
dim_promotion_min = dim_promotion.select("promotion_id", "promotion_type", "discount_pct")

enriched_sales = (
    fact_sales.alias("f")
    .join(broadcast(dim_store_min).alias("s"), "store_id", "left")
    .join(broadcast(dim_product_min).alias("p"), "product_id", "left")
    .join(broadcast(dim_customer_min).alias("c"), "customer_id", "left")
    .join(broadcast(dim_promotion_min).alias("pr"), "promotion_id", "left")
    # Single projection block — Catalyst fuses these into one Project node.
    # `coalesce(discount_pct, 0)` so unpromoted transactions (NULL discount) get full price math.
    .withColumn("gross_revenue", F.round(F.col("quantity") * F.col("unit_price"), 2))
    .withColumn("cost", F.round(F.col("quantity") * F.col("unit_cost"), 2))
    .withColumn(
        "discount_amount",
        F.round(F.col("gross_revenue") * F.coalesce(F.col("discount_pct"), F.lit(0.0)), 2),
    )
    .withColumn("net_revenue", F.round(F.col("gross_revenue") - F.col("discount_amount"), 2))
    .withColumn("gross_margin", F.round(F.col("net_revenue") - F.col("cost"), 2))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Persist as Delta tables
# MAGIC
# MAGIC **`fact_sales` is partitioned by `transaction_date`** (365 daily partitions across CY2024).
# MAGIC Date-range filters in the metrics notebook will partition-prune, reading only the days they need.
# MAGIC
# MAGIC **Perf rules applied to the write:**
# MAGIC
# MAGIC 1. **`format("delta")`** explicitly — Delta gives us partition pruning, file statistics for
# MAGIC    data skipping on non-partition columns, and `OPTIMIZE`/`ZORDER` later if needed.
# MAGIC 2. **`repartition("transaction_date")` before write.** Without this, each of our 200 generation
# MAGIC    tasks would write to all 365 date partitions, producing 200 × 365 = 73,000 small files.
# MAGIC    Repartitioning by `transaction_date` collapses each date to a single task, so the write
# MAGIC    produces ~365 files (one per day) of ~250–300MB each — a healthy file size for Delta.
# MAGIC 3. **`mode("overwrite")` with `overwriteSchema=true`** so re-runs with schema changes don't
# MAGIC    fail on a column drift between iterations of this notebook.
# MAGIC 4. **Dimensions write without partitioning** — they're too small to partition meaningfully.
# MAGIC    A single Delta file per dim is ideal.

# COMMAND ----------

# Dimensions: small, single-file Delta writes. No partitioning, no repartitioning needed.
(dim_store.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("dim_store"))
(dim_product.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("dim_product"))
(dim_customer.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("dim_customer"))
(dim_promotion.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("dim_promotion"))

# fact_sales: repartition by transaction_date BEFORE write to avoid the 200×365 small-files explosion.
# This costs one shuffle, but the shuffle is cheap relative to the file-system cost of writing,
# committing, and later reading 73,000 tiny files.
(
    fact_sales
    .repartition("transaction_date")
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("transaction_date")
    .saveAsTable("fact_sales")
)

# fact_sales_enriched: same partitioning strategy. The metrics notebook reads this table and
# groups by transaction_date / region / category — date partitioning enables pruning for any
# date-bounded metric, and Delta's per-file min/max stats handle the rest.
(
    enriched_sales
    .repartition("transaction_date")
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("transaction_date")
    .saveAsTable("fact_sales_enriched")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Sanity-check displays
# MAGIC
# MAGIC `display()` on a 100M-row Delta table is safe — Databricks samples/limits the preview rather
# MAGIC than materializing the full table to the driver. We deliberately do NOT call `.collect()` or
# MAGIC `.toPandas()` anywhere in this notebook.

# COMMAND ----------

display(spark.table("dim_store"))
display(spark.table("dim_product"))
display(spark.table("dim_customer"))
display(spark.table("dim_promotion"))
display(spark.table("fact_sales"))
display(spark.table("fact_sales_enriched"))
