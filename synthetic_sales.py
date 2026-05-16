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
# Three dims (store, product, promotion) are intentionally tiny — they fit under the
# broadcast-join threshold (default spark.sql.autoBroadcastJoinThreshold = 10MB) and become
# map-side broadcast joins with zero shuffle on the fact side.
# dim_customer is the EXCEPTION: at 200M rows × ~30 bytes ≈ 6GB serialized, it's far too
# large to broadcast. Any fact->customer join must use SortMergeJoin (both sides shuffled
# by customer_id). See section 6 for the strategy change and metrics.ipynb for a demo.
NUM_STORES = 200
NUM_PRODUCTS = 5_000
NUM_CUSTOMERS = 200_000_000
NUM_PROMOTIONS = 50

# Partition count for dim_customer generation: 200M / 500K per partition = 400. Same sizing
# rationale as the fact table — partitions large enough to amortize task overhead, small
# enough to avoid GC pressure.
CUSTOMER_NUM_PARTITIONS = 400

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
# MAGIC ## 3. `dim_customer` — 200M rows (NOT broadcastable)
# MAGIC
# MAGIC `signup_date` is built by adding a deterministic day offset (0–2919, ~8 years) to a base date.
# MAGIC
# MAGIC **Perf at 200M-row scale:**
# MAGIC
# MAGIC - **No broadcast.** At ~30 bytes/row × 200M rows ≈ 6GB serialized, this dim is ~600x the
# MAGIC   default broadcast threshold (10MB). Forcing `broadcast()` here would OOM the driver during
# MAGIC   the broadcast `collect`, then OOM every executor that tried to hold the hash table.
# MAGIC - **Partitioned generation.** 400 partitions ≈ 500K rows/task — same sweet spot as the fact.
# MAGIC - **Bucketing on `customer_id`** (see write-out below). This is the key optimization for the
# MAGIC   fact↔customer join: bucketing both tables on the join key eliminates the customer-side
# MAGIC   shuffle of a SortMergeJoin, since matching keys are already co-located by file.
# MAGIC
# MAGIC **Why generation cost matters less than join cost:** generation is a single linear stage —
# MAGIC 400 narrow tasks compute and write in parallel, no shuffle. The expensive operation downstream
# MAGIC is the join, which is where bucketing earns its keep.

# COMMAND ----------

dim_customer = (
    spark.range(1, NUM_CUSTOMERS + 1, step=1, numPartitions=CUSTOMER_NUM_PARTITIONS)
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
# MAGIC - **`store_id` is DELIBERATELY SKEWED.** ~50% of transactions land on `store_id = 1` (a "flagship"
# MAGIC   hot key); the remaining ~50% are uniformly distributed across stores 2..200. This simulates a
# MAGIC   real retail pattern (one mega-store dominating volume) and gives the metrics notebook a
# MAGIC   concrete case study for skew detection and salted aggregation. Without this, every store would
# MAGIC   have ~0.5% of rows and skew-handling techniques would be unmotivated.
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
    # store_id: deliberately skewed. With prob ~0.5 the row goes to store_id=1 (the hot key);
    # otherwise it's uniformly distributed across stores 2..NUM_STORES. The seeded rand() makes
    # the skew reproducible across runs. Result: ~50M rows on store 1, ~50M rows spread across
    # 199 other stores (~250K each). This is the workload that motivates salted aggregation
    # in the metrics notebook.
    .withColumn(
        "store_id",
        F.when(
            F.rand(seed=77) < F.lit(0.5),
            F.lit(1).cast("long"),
        ).otherwise(
            (F.pmod(F.hash(F.col("transaction_id"), F.lit("s")), F.lit(NUM_STORES - 1)) + F.lit(2)).cast("long")
        ),
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
# MAGIC 2. **Mixed join strategies — broadcast for tiny dims, SortMergeJoin for `dim_customer`.**
# MAGIC    `dim_store` / `dim_product` / `dim_promotion` are all <1MB → `broadcast()` hints force a
# MAGIC    BroadcastHashJoin (no fact-side shuffle). `dim_customer` at 200M rows ≈ 6GB CANNOT be
# MAGIC    broadcast — we'd OOM the driver collect and the executor heap. Instead we let Catalyst
# MAGIC    pick SortMergeJoin (both sides shuffled and sorted by `customer_id`). Crucially, both
# MAGIC    `fact_sales` and `dim_customer` are **bucketed on `customer_id`** at write time (see
# MAGIC    section 7), which lets the SortMergeJoin skip the shuffle/sort phase entirely on subsequent
# MAGIC    reads — files already co-locate matching keys.
# MAGIC 3. **The defensive `broadcast()` hint logic still applies for the small dims.** If a future
# MAGIC    change pushes one of them past the auto-broadcast threshold, the plan would silently
# MAGIC    degrade to a SortMergeJoin (full shuffle of the 100M-row fact). The hint forces a broadcast
# MAGIC    hash join and surfaces the problem as an OOM instead of silent slowdown. We deliberately
# MAGIC    do NOT broadcast `dim_customer` — it is *known* to be too large.
# MAGIC 4. **`LEFT` joins.** Defensive — keep every fact row even if a key is unexpectedly missing.
# MAGIC    Only `promotion_id` is legitimately nullable.
# MAGIC 5. **All numeric columns derived in one pass.** `gross_revenue`, `discount_amount`,
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
    # NO broadcast() on dim_customer — it's 6GB. Catalyst will pick SortMergeJoin here.
    # Once both sides are read from their bucketed-on-customer_id Delta tables (section 7),
    # the SMJ can use the bucketing to skip the shuffle phase: matching keys are already
    # co-located by file, so the join becomes a streaming sort-merge across pre-sorted partitions.
    .join(dim_customer_min.alias("c"), "customer_id", "left")
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
# MAGIC **`fact_sales` is partitioned by `transaction_date`** (365 daily partitions across CY2024)
# MAGIC AND **bucketed by `customer_id`** into `NUM_BUCKETS` files per date partition. `dim_customer`
# MAGIC is bucketed identically. This is the key change driven by the 200M-row dim_customer.
# MAGIC
# MAGIC **Why bucket on `customer_id`:** the fact↔dim_customer SortMergeJoin would otherwise need to
# MAGIC shuffle BOTH the 100M-row fact AND the 200M-row dim by `customer_id` on every read — that's
# MAGIC ~10–15GB of shuffle for the fact + ~6GB for the dim, every single query. Bucketing both
# MAGIC tables on the same key, into the same number of buckets, with the same hash function, lets
# MAGIC Spark recognize at plan time that matching keys are already co-located. The join plan changes
# MAGIC from `Exchange → Sort → SortMergeJoin → Exchange → Sort` (per side) to just `SortMergeJoin`
# MAGIC over pre-bucketed inputs — no shuffle, no sort.
# MAGIC
# MAGIC **Choosing `NUM_BUCKETS`:** rule of thumb is bucket count × avg file size ≈ healthy file size
# MAGIC (~256MB). 200M customer rows × ~30 bytes = 6GB → 6GB / 256MB ≈ 24 buckets. We round to 32
# MAGIC (a power of two helps the hash partitioner spread evenly). Both tables MUST use the same
# MAGIC bucket count or Spark falls back to a re-shuffle.
# MAGIC
# MAGIC **Other perf rules:**
# MAGIC
# MAGIC 1. **`format("delta")`** explicitly — Delta gives us partition pruning, file statistics for
# MAGIC    data skipping on non-partition columns, and `OPTIMIZE`/`ZORDER` later if needed.
# MAGIC 2. **`repartition("transaction_date", "customer_id")` before write.** With bucketing layered
# MAGIC    on top of date partitioning, we want each writer task to handle exactly one
# MAGIC    `(transaction_date, bucket)` slot. Repartitioning by both columns aligns the in-memory
# MAGIC    layout with the on-disk layout and avoids the small-files explosion.
# MAGIC 3. **`mode("overwrite")` with `overwriteSchema=true`** so re-runs with schema changes don't
# MAGIC    fail on a column drift between iterations of this notebook.
# MAGIC 4. **Small dims (`dim_store`, `dim_product`, `dim_promotion`) write without partitioning or
# MAGIC    bucketing** — they're tiny enough to broadcast at query time. Bucketing them would be
# MAGIC    pure overhead.

# COMMAND ----------

# DBTITLE 1,Bucketing configuration for customer_id co-location
# Both fact_sales and dim_customer use the same bucket count on customer_id so SortMergeJoin
# can skip the shuffle phase on read. Changing this value requires re-writing BOTH tables.
NUM_BUCKETS = 32

# COMMAND ----------

# Small dims: single-file Delta writes. No partitioning, no bucketing needed.
(dim_store.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("dim_store"))
(dim_product.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("dim_product"))
(dim_promotion.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("dim_promotion"))

# dim_customer: 200M rows, bucketed on customer_id so fact↔customer SMJ can skip the shuffle.
# Note: Spark's bucketBy is only respected on the Hive metastore path (saveAsTable), not on
# path-based saves — that's why we use saveAsTable here even though Delta is the file format.
# Repartitioning by customer_id before the write aligns the in-memory layout with the bucket
# layout, so each writer task produces exactly one bucket file (no scatter writes).
(
    dim_customer
    .repartition(NUM_BUCKETS, "customer_id")
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .bucketBy(NUM_BUCKETS, "customer_id")
    .sortBy("customer_id")  # within-bucket sort lets SMJ stream pre-sorted partitions
    .saveAsTable("dim_customer")
)

# fact_sales: partition by transaction_date AND bucket by customer_id.
# Layered partitioning + bucketing: 365 date partitions × 32 customer buckets = 11,680 logical
# slots. Each slot becomes one Delta file of ~30-40MB (100M rows × ~50 bytes / 11,680). That's
# on the small side for Delta, but the trade-off is worth it: the alternative is a per-query
# shuffle of 5-10GB on the customer_id join.
#
# Repartitioning by (transaction_date, customer_id) before the write aligns the in-memory
# layout with the on-disk layout — each task writes to exactly one (date, bucket) slot.
(
    fact_sales
    .repartition("transaction_date", "customer_id")
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("transaction_date")
    .bucketBy(NUM_BUCKETS, "customer_id")
    .sortBy("customer_id")
    .saveAsTable("fact_sales")
)

# fact_sales_enriched: same partition + bucket strategy as fact_sales. The metrics notebook
# reads this table and groups by transaction_date / region / category / loyalty_tier — date
# partitioning enables pruning for date-bounded metrics, customer_id bucketing means any
# downstream join back to dim_customer (or to another customer-keyed table) skips the shuffle.
(
    enriched_sales
    .repartition("transaction_date", "customer_id")
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("transaction_date")
    .bucketBy(NUM_BUCKETS, "customer_id")
    .sortBy("customer_id")
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
