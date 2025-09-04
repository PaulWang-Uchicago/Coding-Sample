# Assignment 3
**Name:** Paul Wang  
**ID:** zw2685

---

## Question 1 Exploratory Data Analysis

### Data Loading & Preliminary Inspection
- **File:** [EDA_NYCTaxi.ipynb](https://github.com/macs30123-s25/a3-PaulWang-Uchicago/blob/main/EDA_NYCTaxi.ipynb)

Loads the 2015 NYC taxi Parquet files from S3 into a Dask DataFrame. Performs schema validation and computes basic summary statistics (mean, median, quartiles) for key numeric fields.

#### Interpretation of Categorical Feature Cardinality

The output shows the number of cardinality for selected categorical features in the NYC Taxi dataset:

| Feature             | Cardinality | Interpretation                                                                                          |
|---------------------|-------------|----------------------------------------------------------------------------------------------------------|
| `payment_type`      | 5           | There are 5 distinct payment methods (e.g., credit card, cash, no charge). Suitable for one-hot encoding. |
| `RatecodeID`        | 7           | Represents 7 fare rate codes including standard, JFK, Newark, etc. Suitable for one-hot encoding.       |
| `store_and_fwd_flag`| 2           | Binary flag (`Y` or `N`) indicating if the trip record was stored and forwarded.                        |
| `PULocationID`      | 264         | 264 unique pickup location zones. High-cardinality; may require dimensionality reduction.               |
| `DOLocationID`      | 264         | 264 unique dropoff location zones. Also high-cardinality; treat similarly to pickup zones.              |

---

### Visualization Overview
I organized my visualization suite to uncover the key dimensions driving fares, tips, and demand—moving from the granular to the aggregate:

- Micro-level (Trip) Patterns:
I start by examining how core variables interact (e.g. distance × passenger count; tip % vs. distance) so I know where to introduce interaction terms and non-linear transformations.

- Spatial Structure:
I map the busiest origin–destination corridors to pinpoint categorical features (e.g. airport flags, zone IDs) that capture location-specific tipping or demand behavior.

- Macro-level (Temporal & Aggregate) Dynamics:
I chart hourly and daily tipping patterns alongside revenue or surcharge trends to identify cyclical (rush-hour vs. weekend vs. season) and holiday effects, informing my time-based and calendar features.

By drilling into distributional, spatial, and then temporal “slices” of the data, I ensure each visualization directly translates into tailored features for my tip-prediction and demand models.

---

### Visualization 1: Trip Distance by Passenger Count
I compare the distance distributions for parties of 1–4 passengers. Solo riders show the widest range (up to 10 mi), while larger groups concentrate in short‐hop trips.

![trip_distance.png](https://github.com/macs30123-s25/a3-PaulWang-Uchicago/blob/main/trip_distance.png)

#### Key Patterns in visualization 1

1. **Passengers = 1**  
   The top‐left panel shows a classic long‐tailed distribution: most one-passenger trips are under 3 miles (with a peak around 1 mile), but there remains a substantial tail of longer journeys up to 10 miles. This reflects the high volume of quick city hops by individuals, mixed with occasional longer point‐to‐point rides.

2. **Passengers = 2**  
   The top‐right panel likewise peaks between 1–2 miles, but the overall volume is lower than for solo riders. This suggests that two-person trips are still mostly short runs (e.g. couples or friends), with fewer long-distance journeys.

3. **Passengers = 3**  
   In the bottom‐left, the distribution is markedly narrower: nearly all three-passenger trips fall between 0.5 and 3 miles, with very few beyond 4 miles. Larger parties appear to avoid long rides—perhaps due to higher absolute fares or logistical constraints favoring shorter, local trips.

4. **Passengers = 4**  
   The bottom-right shows the smallest volume and the tightest clustering around 1–2 miles. Four-passenger bookings are rare and almost exclusively short distances, indicating that large groups predominantly use taxis for short legs (e.g. airport shuttles, group outings in neighboring areas).

Passenger count and trip distance are clearly interdependent: single riders undertake the widest variety of trip lengths, while larger parties are concentrated on shorter hops. Including **interaction terms** (e.g. `distance × passenger_count`) will allow the model to adjust expectations—predicting, for instance, that a four-passenger ride is unlikely to exceed 4 miles, or that large groups tip differently on shorter versus longer routes. This stratification can sharpen distance‐based features and improve tip‐prediction accuracy.

---

### Visualization 2: Tip % & Average Fare vs. Distance
This dual‐axis line chart reveals that fare grows almost linearly with distance, whereas tip percentage peaks around 9–10mi before declining on very long trips.

![tip_pct_fare.png](https://github.com/macs30123-s25/a3-PaulWang-Uchicago/blob/main/tip_pct_fare.png)

#### Key Patterns in Visualization 2

1. **Average Fare Rises Steadily**  
   The yellow line shows that mean fare increases almost linearly with distance, from around \$10 for a 1-mile trip up to nearly \$180 at 50 miles. There is a slight uptick in slope beyond 30 miles, likely reflecting additional tolls or long-haul surcharges on extended journeys.

2. **Non-Monotonic Tip Percentage**  
   The teal line (mean tip %) does not simply decline as distance grows. Instead, tip rates climb from about 12.5% at very short distances to a peak of roughly 15.5% around 9–10 miles, then fall to about 10% near 22 miles. There’s a smaller rebound up to about 13% at about 30 miles before a steady drop for the longest trips, ending around 7.3% at 50 miles.

3. **Mid-Distance Generosity Bump**  
   The pronounced bump in tip percentage for mid-length trips (8–12 miles) suggests that rides of moderate length encourage especially generous tipping, even though the absolute fare is still moderate.

4. **Tip Fatigue on Very Long Trips**  
   Beyond 35 miles, tip rates taper off sharply despite very high fares. This “tip fatigue” could reflect passenger reluctance to leave a large absolute gratuity on an already expensive fare.

The non‐monotonic relationship between trip distance and tip percentage suggests that distance should be modeled nonlinearly (e.g. by introducing piecewise bins) so that the model can learn the mid-distance generosity bump and the long-haul decline. Moreover, because average fare grows almost linearly with distance while tip percentage follows a distinct curve, including an interaction between fare and distance (or a derived feature such as `fare/distance`) will help the model differentiate city-run tipping norms from long-haul behaviors.

---

### Visualization 3: Top 15 Pickup - Dropoff Zones Heatmap
A heatmap of the busiest origin–destination pairs spotlights two dominant airport corridors (PULocationID 237 - 236) and the Midtown - LGA flow.

![od_matrix.png](https://github.com/macs30123-s25/a3-PaulWang-Uchicago/blob/main/od_matrix.png)

#### Key Patterns in Visualization 3

1. **Airport - City Dominance**  
   The darkest cells, particularly at PULocationID 237 - DOLocationID 236, represent over half a million rides between the two main airports. These long‐haul trips carry flat‐rate fees, higher tolls, and travel‐oriented passengers, all of which tend to drive higher tips and distinct tip‐percentage distributions compared to city‐center hops.

2. **Midtown - LaGuardia (LGA) Corridor**  
   The next‐brightest block appears around 161 - 230, corresponding to Midtown Manhattan - LGA. With about 300,000 trips, this flow similarly exhibits longer distances, extra surcharges, and different passenger profiles, making it another critical feature for modeling tipping behavior.

3. **High‐Density Intra-Zone Trips**  
   Several moderate intensity diagonal cells (e.g. zone 48 - 48 or 68 - 68) highlight numerous very short, within-neighborhood rides in dense areas. These trips usually have minimal tolls and low durations, which correlates with lower tip amounts and tighter tip‐rate distributions.

To improve tip‐prediction accuracy, it’s important to include both pickup and dropoff zone identifiers so the model can learn the unique tipping patterns associated with each corridor—whether it’s a short neighborhood hop or a long airport transfer. Adding a simple binary flag such as `is_airport_trip` (true when either the pickup or dropoff zone corresponds to an airport) further sharpens this distinction, since airport rides not only cover greater distances and incur extra fees but also involve travelers who tip differently.

---

### Visualization 4: Hourly Share of Trips by Weekday
Normalizing each weekday’s hourly volumes shows clear commuter peaks around 8–10 AM and 5–7 PM, with weekend demand shifted later into the evening.

![hourly_share.png](https://github.com/macs30123-s25/a3-PaulWang-Uchicago/blob/main/hourly_share.png)

#### Key Patterns in Visualization 4

The most striking pattern is the twin peaks of weekday demand: a sharp climb beginning around 6 AM that culminates in a morning rush (8–10 AM), followed by a modest midday plateau and then an even larger evening surge between 5 PM and 7 PM. During those peak windows, each hour accounts for 5–7% of that day’s trips, whereas the overnight hours (2–5 AM) drop to under 1%. Also, Friday shows a slightly higher share during the morning (over 3% at 8 AM) and a later, more drawn‐out evening peak than Monday–Thursday.

Weekend patterns diverge significantly. On Saturday, volumes ramp up earlier around 9 AM, and the evening peak is flatter but sustained from 4 PM through midnight, with each hour in that block contributing 5–6% of daily rides. Sunday begins the day with the highest single‐hour share (over 6% at midnight) but then settles into a broader midday peak (10 AM–2 PM) before tapering off steadily. These differences underscore how leisure travel reshapes the timing of demand compared to the commuter‐driven rhythms of the workweek.

These patterns show that hour‐of‐day and day‐of‐week interact in non‐trivial ways, so any tip‐prediction or demand model should include both features to capture the twin morning and evening peaks versus the flat early‐morning trough. A simple `is_weekend` flag remains essential to distinguish the leisure‐driven Saturday/Sunday timing from weekday commutes, and binary `morning_rush` (7–10 AM) and `evening_rush` (5–8 PM) indicators can directly encode the hours with systematically higher volumes.

---

### Visualization 5: Average Tip% by Day & Hour
Overlaying tip percentages on the same 7×24 grid uncovers pronounced evening spikes and uniformly lower weekend tips.

![tip_pct_heatmap.png](https://github.com/macs30123-s25/a3-PaulWang-Uchicago/blob/main/tip_pct_heatmap.png)

#### Key Patterns in Visualization 5

1. **Clear Weekday Evening Peak**  
   Across Monday – Friday, tip percentages are lowest in the early morning (around 6–8 AM) and then steadily climb through the day, reaching their highest values between 4 PM and 7 PM (peaking above 23 % on Tuesday at 4 PM). This “rush-hour” surge likely reflects higher‐fare, longer rides (e.g. commuters heading home) and passengers tipping more generously during busy evening periods.

2. **Midday Plateau and Overnight Trough**  
   Between about 10 AM and 2 PM on weekdays, tip rates hover around 20-22%, suggesting a stable lunchtime pattern. After 8 PM the rate gradually falls back toward 21%, reaching its overnight trough again just before dawn.

3. **Muted Weekend Dynamics**  
   Saturday and Sunday show a much flatter profile: average tips stay in the 21 – 22% range all day, with only a slight bump around midday on Saturday. Weekend evenings don’t exhibit the sharp “rush‐hour” peak seen on weekdays. Tipping behavior is more uniform, perhaps reflecting leisure travel rather than commuter flows.

4. **Day Variations**  
   - Tuesday shows the single highest tip rate of the week, possibly driven by mid-week business travel or consistently heavier evening traffic.  
   - Friday evening rates (5 – 7 PM) are slightly lower than mid-week peaks, hinting at different rider demographics or more casual tipping on the last workday.  
   - Sunday remains the lowest of all days, again reflecting weekend leisure patterns.


Accurate tip modeling requires capturing the joint effect of both the day of week and the hour of day, since weekday evenings exhibit pronounced spikes in tipping that neither feature alone can explain. Introducing a simple `is_weekend` flag helps the model distinguish the relatively flat, uniform tipping patterns seen on Saturdays and Sundays from the sharper peaks of the workweek. Likewise, a binary `rush_hour` indicator (e.g. trips beginning between 4 PM and 7 PM) directly encodes the period of systematically higher tips, sparing the model from having to learn the precise shape of that surge. 

---

### Visualization 6: Daily Toll Revenue
Daily toll sums climb from \$80K to \$160K through spring, dip in summer, then rebound in fall.

![tolls_trend.png](https://github.com/macs30123-s25/a3-PaulWang-Uchicago/blob/main/tolls_trend.png)

#### Key Patterns Visualization 6

The blue line shows that total toll revenue climbed steadily over the first half of 2015—from roughly \$80K/day in January to a peak around \$160K/day in May and June—reflecting both rising ride volumes and cumulative toll increases. After mid-year, there is a noticeable summer dip: daily tolls ease back into the \$100K–\$130K range through July and August, likely driven by seasonal shifts in demand. 

From September through November, toll revenue recovers to the \$130K–\$150K band, before dipping again around major holidays (e.g. early July, late November) and then settling into a moderately high plateau into December. The 30-day rolling average (not pictured here) smooths out the daily volatility, highlighting a clear upward trend into late spring, a flattening over the summer months, and a modest rebound in the fall.  

The pronounced seasonal and holiday‐driven swings in daily toll revenue imply that the predictive model should explicitly account for both long‐term trends and calendar‐based effects. For example, incorporating cyclic features such as month‐of‐year or day‐of‐year (via sine/cosine transforms) will capture the steady rise into late spring, the summer lull, and the fall rebound, while a separate linear or piecewise time index can model any underlying upward trend in toll levels. I also want to add binary flags for major holidays or special events to handle the sharp dips around dates like July 4th and Thanksgiving.

---

### Visualization 7: Weekly Extra Charges vs. MTA Tax
Extra surcharges and the flat MTA taxi tax track nearly in lockstep (around \$ 1 M/week), implying both serve as proxies for overall volume.

![weekly_totals.png](https://github.com/macs30123-s25/a3-PaulWang-Uchicago/blob/main/weekly_totals.png)

#### Key Patterns in Visualization 7

Over 2015, the total weekly extra charges (blue) and MTA tax revenue (orange) track almost perfectly in parallel, rising sharply from roughly \$0.4M in January to just over \$1M per week by March. Both series fell back into the \$0.8M–\$0.9M range around July to August, before recovering into the \$0.9M–\$1.0M band through the fall. The close alignment indicates that extra surcharges scale very directly with the flat MTA taxi tax, suggesting both are driven primarily by overall trip volume. Finally, both series dip steeply in late December, reflecting the holiday slowdown in taxi activity.  

The near‐perfect parallelism of weekly extra charges and MTA tax revenue suggests both series are really acting as proxies for overall taxi volume, so including a single `weekly surcharge` variable in the models can help capture fluctuations in demand without introducing multicollinearity. To account for the pronounced summer lull and holiday droughts, add a week‐of‐year cyclic feature and binary holiday/season flags that mark major slowdowns like late December.

---

## Question 2 Implementing a Reproducible Machine Learning Pipeline

For this question, PySpark is used in an EMR Notebook to extend the linear regression model trained in class by engineering new features, assembling them into a reusable pipeline, and evaluating the model performance. 

### (a) Feature Engineering
- **File:** [ML_NYCTaxi.ipynb](https://github.com/macs30123-s25/a3-PaulWang-Uchicago/blob/main/ML_NYCTaxi.ipynb)

Below is an updated list of the features added to the PySpark tip‐prediction pipeline:

1. **corridor_id**  
   A concatenation of `PULocationID` and `DOLocationID` (e.g. `"161_230"`) to capture specific origin→destination flows.

2. **is_airport_trip**  
   Binary flag (0/1) indicating whether either pickup or dropoff zone is an airport (zones 236 or 237).

3. **corridor_ohe**  
   One-hot encoding of `corridor_id` via `StringIndexer` + `OneHotEncoder` to represent the top corridors as categorical features.

4. **hour**  
   Integer hour of pickup (0–23).

5. **weekday**  
   Integer day of week (1=Monday … 7=Sunday).

6. **is_weekend**  
   Binary flag (0/1) for Saturday or Sunday.

7. **rush_hour**  
   Binary flag (0/1) for evening rush (16–19h).

8. **dist_bin**  
   Bucketized `trip_distance` into 4 bins (`(-∞,2], (2,8], (8,15], (15,∞)`) for non-linear distance effects.

9. **fare_per_mile**  
   Floating-point feature = `fare_amount` ÷ `trip_distance`, capturing average price per mile.

10. **dist_times_pax**  
    Interaction term = `trip_distance` × `passenger_count`, modeling combined distance‐party effects.

11. **pax_ohe**  
    One-hot encoding of `passenger_count` (1–4) to treat group sizes as categorical inputs.

First, I identify two sets of special zones—airport and midtown—by hard-coding their TLC LocationIDs (`[236, 237]` for JFK/LGA and `[161, 162, 163, 186]` for Midtown). Trips touching these areas tend to follow very different distance, surcharge, and tipping behaviors than the rest of the city, so it makes sense to mark them out explicitly.

To further isolate airport behavior, I add a binary `is_airport_trip` flag that is set to 1 whenever either endpoint is in an airport zone. In a similar spirit, `is_midtown_trip` flags journeys that begin or end in the dense Midtown taxi network. Both of these broad-strokes indicators pick up on the distinct fare structures, demand surges, and tipping norms typical of airport shuttles and Midtown short hops.

```python
# 1) Define any special zone lists
airport_zones = [236, 237]               
midtown_zones = [161, 162, 163, 186]     

# 2) Build corridor_id and the two binary flags
data = (data
    # origin–destination corridor as “PU_DO”
    .withColumn("corridor_id",
        F.concat_ws("_", F.col("PULocationID"), F.col("DOLocationID"))
    )
    # airport‐trip flag
    .withColumn("is_airport_trip",
        F.when(
            F.col("PULocationID").isin(airport_zones) |
            F.col("DOLocationID").isin(airport_zones),
            1
        ).otherwise(0)
    )
    # midtown‐trip flag
    .withColumn("is_midtown_trip",
        F.when(
            F.col("PULocationID").isin(midtown_zones) |
            F.col("DOLocationID").isin(midtown_zones),
            1
        ).otherwise(0)
    )
)
```

Next, I build a new `corridor_id` string by concatenating the pickup and dropoff zone IDs (for example, `"142_236"`). This single categorical variable captures origin → destination pairs in one fell swoop, allowing the regression to learn separate coefficients for common flows.

Because `corridor_id` can take on hundreds of unique values, I convert it into a numeric form that my ML pipeline can digest: first with a `StringIndexer` (assigning each corridor a stable integer index), and then with a `OneHotEncoder` to turn that index into a sparse binary vector. This approach gives each high-volume OD pair its own parameter, while exploiting sparsity to keep computation tractable.

```python
corridor_indexer = StringIndexer(
    inputCol  = "corridor_id",
    outputCol = "corridor_idx",
    handleInvalid="keep"
)

corridor_encoder = OneHotEncoder(
    inputCols  = ["corridor_idx"],
    outputCols = ["corridor_ohe"]
)
```

To capture the strong temporal rhythms in tipping behavior, I begin by extracting relevant time-based features from the original `tpep_pickup_datetime` column. I rename this timestamp to `pickup_ts` for clarity and ease of use in subsequent transformations.

From this timestamp, I derive the hour of day using the `hour()` function and cast it to an integer. This `hour` feature is useful for identifying daily cycles, such as morning or evening rush periods when tipping may spike due to higher demand or driver scarcity.

I also extract the day of week using `dayofweek()`, which returns values ranging from 1 (Sunday) to 7 (Saturday). However, this format is not aligned with the ISO standard, which defines Monday as 1 and Sunday as 7. To resolve this, I re-index the day values using modular arithmetic to produce a `weekday_iso` column that conforms to the ISO standard. This lets me consistently identify weekdays and weekends in downstream analyses.

Building on this, I define an `is_weekend` binary feature that is set to 1 if the trip occurred on a Saturday or Sunday. Since tipping norms and rider behavior often shift between weekends and weekdays, this indicator helps the model adjust expectations accordingly.

Finally, I introduce a `rush_hour` flag for trips that occur during the late afternoon peak (4–7 PM). This feature signals times of higher congestion, limited driver supply, and potentially longer trip durations—all of which may influence fare size and tipping propensity. These temporal features allow the model to capture predictable, time-linked variation in tipping behavior without needing to memorize individual timestamps.

```python
data = (
    data
    .withColumn("pickup_ts", F.col("tpep_pickup_datetime"))
    .withColumn("hour",       F.hour("pickup_ts").cast("int"))
    .withColumn("dow_sun1",   F.dayofweek("pickup_ts").cast("int"))
    .withColumn(
        "weekday_iso",
        ((F.col("dow_sun1") + F.lit(5)) % F.lit(7) + F.lit(1)).cast("int")
    )
    .withColumn("is_weekend", (F.col("weekday_iso") >= 6).cast("int"))
    .withColumn("rush_hour",  F.when(F.col("hour").between(16,19), 1).otherwise(0))
)
```

I introduce non‐linear representations of trip distance to allow the model to learn different tipping patterns for short, mid‐range, long, and very‐long trips. First, I define `distance_splits` to separate the continuous `trip_distance` into four intuitive buckets—short (≤2 mi), mid (2–8 mi), long (8–15 mi), and very‐long (>15 mi)—and apply Spark’s `Bucketizer` to generate a new categorical feature `dist_bin`. This bucketization captures the “sweet spot” in the mid‐range where riders often tip more generously, without forcing a purely linear assumption.

Next, I compute a `fare_per_mile` feature by dividing the total fare by the trip distance (when distance is positive) and defaulting to zero otherwise. This ratio conveys how expensive each mile of the trip was, helping the model distinguish, for example, short rides with high per‐mile fares (like airport trips) from long trips with lower unit costs. Because fare structure and rider generosity can vary markedly by trip economy, this normalized feature adds valuable granularity.

Finally, I create an interaction term `dist_times_pax` by multiplying the raw trip distance by `passenger_count`. This captures how group size compounds distance effects. For instance, multi‐passenger rides may behave differently over the same distance than solo trips. By including this floating‐point interaction, I ensure the model can adjust its distance sensitivity based on party size rather than treating every mile as “worth” the same tipping potential regardless of how many people are on board.

```python
# Bucketize trip_distance (non-linear bins) to categorical buckets
distance_splits = [-float("inf"), 2, 8, 15, float("inf")]   # short, mid, long, very-long
bucketizer = Bucketizer(
    splits=distance_splits,
    inputCol="trip_distance",
    outputCol="dist_bin")                                  

# fare per mile
data = data.withColumn(
    "fare_per_mile",
    F.when(F.col("trip_distance") > 0,
           F.col("fare_amount") / F.col("trip_distance")
    ).otherwise(0.0)
)

# interaction: distance × passenger_count  (integer × float)
data = data.withColumn(
    "dist_times_pax",
    F.col("trip_distance") * F.col("passenger_count")
)
```

```python
data = data.withColumn(
    "passenger_count_dbl",
    F.col("passenger_count").cast("double")
)
```

I convert the continuous `passenger_count_dbl` into a categorical representation so that the model can learn distinct tipping behaviors across different party sizes. First, I apply Spark’s `StringIndexer` to map each unique passenger count (as a double) to an integer index in the new column `pax_idx`, ensuring that any unexpected or missing values are handled gracefully by keeping them as a separate category. Then, I use `OneHotEncoder` on `pax_idx` to produce a sparse vector `pax_ohe`, which allows the linear regression to assign an independent coefficient to each party‐size category rather than treating passenger count as a simple numeric input. This one‐hot encoding captures non‐linear shifts in tipping patterns between single riders, couples, and larger groups without imposing an arbitrary ordering.  

```python
pax_indexer = StringIndexer(
    inputCol="passenger_count_dbl",
    outputCol="pax_idx",
    handleInvalid="keep" 
)
pax_encoder = OneHotEncoder(
    inputCols=["pax_idx"],
    outputCols=["pax_ohe"]
)
```

In this step, I bring together all of the engineered columns into a single feature vector that can be consumed by the regression algorithm. I list out the untransformed numeric and binary flags—such as the cyclic hour (`hour`), weekend indicator (`is_weekend`), and rush‐hour flag (`rush_hour`)—alongside the continuous variables that capture distance and fare dynamics (`fare_per_mile`, `dist_times_pax`). I also include the spatial indicator (`is_airport_trip`, `is_midtown_trip`), the one‐hot encoded passenger‐count vector (`pax_ohe`), and the bucketized distance category (`dist_bin`) to allow the model to learn non‐linear distance effects. Finally, I apply `VectorAssembler` to these columns, producing a new column called `features` that contains a single vector of all predictors. This consolidated vector simplifies downstream model training by presenting every engineered input in a uniform format.  

```python
feature_cols = [
    # temporal / binary
    "hour", "is_weekend", "rush_hour",

    # floating-point interactions
    "fare_per_mile", "dist_times_pax",

    # spatial flags
    "is_airport_trip", "is_midtown_trip",

    # nonlinear distance bucket
    "dist_bin",

    # high-cardinality OD pairs
    "corridor_ohe",

    # party-size one-hot
    "pax_ohe",
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)
```

### (b) Pipeline Construction
I instantiate my core learner: a regularized linear regression that predicts tip_amount from the assembled feature vector. I set a modest number of iterations and apply both L1 and L2 penalties (elasticNetParam=0.2) to help control overfitting.

With the learner defined, I assemble my full modeling pipeline. This chains together all of my feature transformers—indexing and encoding the passenger counts, bucketizing distances, vectorizing features—followed by the regression stage. Now I have one object that encapsulates every step from raw DataFrame to trained model.

Before I even tune anything, I split the data once into an 80/20 train/test hold-out. This ensures that after cross-validation and model selection on the training partition, I still have an untouched subset on which to report a final, unbiased performance estimate.

```python
# 1) Define the regression learner
lr = LinearRegression(
    featuresCol="features",
    labelCol="tip_amount",
    maxIter=50,
    regParam=0.1,
    elasticNetParam=0.2
)

# 2) Build the full pipeline
pipeline = Pipeline(stages=[
    corridor_indexer, corridor_encoder,
    pax_indexer, pax_encoder,
    bucketizer,
    assembler,
    lr
])

# 3) Split into train / test once
train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)

# 4) Define evaluator
evaluator = RegressionEvaluator(
    labelCol="tip_amount",
    predictionCol="prediction"
)
```

### (c)
When I define a sequence of Transformers and Estimators in a Spark Pipeline, Spark does not immediately read or transform any data. Instead, each .withColumn, StringIndexer, Bucketizer, VectorAssembler, and so on simply adds a node to an internal logical plan DAG (directed acyclic graph) representing the computation I intend to perform. No actual I/O or computation happens until I invoke an action—such as pipeline.fit(), model.transform(), or writing out results—which triggers Catalyst’s optimizer to compile the logical plan into a physical execution plan and then schedule tasks across the cluster. At that point, Spark reads the input Parquet files, applies all of the transformations in the most efficient order (including predicate pushdown, projection pruning, and pipelined execution), and materializes the results.

Dask’s execution model is conceptually similar in that it also constructs a task graph lazily when I invoke Dask DataFrame or Array operations, but it tends to be more fine-grained: tasks correspond to small blocks of data and Dask doesn’t have a central query optimizer like Catalyst. Dask only executes when I call .compute() (or write out data), at which point its scheduler breaks the graph into tasks and runs them—either in threads, processes, or distributed workers. Whereas Spark’s DAG optimizer can reorder, fuse, and optimize whole-job pipelines before any data moves, Dask relies on smaller task graphs with less global optimization, trading off some overhead for more flexibility and direct control over parallelism.

---

## Question 3

After validating the ML pipeline from Question 2. I run my model on a larger cluster (1 primary + 7 core m5.xlarge nodes on EMR), a 5-fold cross-validated grid search is performed to identify optimal regularization parameters and evaluate model performance.

I instantiate a CrossValidator, pointing it at my full pipeline (which already bundles all of my feature‐engineering steps and the linear regression estimator) and the evaluator that knows how to compute RMSE.  By setting numFolds=3, I ask Spark to perform three‐fold cross validation, and with parallelism=2 I enable two models to be trained in parallel, speeding up the search.

Next, I call crossval.fit(train_df), which kicks off the cross‐validation process.  Under the hood, Spark will split train_df into three folds, train on two of them and validate on the third, rotating through all combinations, and will track which combination of hyperparameters (if I’d provided any) yields the lowest validation RMSE.  The result, cvModel, contains both the full history of models and the single best pipeline.

Finally, I extract cvModel.bestModel (the pipeline instance that achieved the lowest cross‐validation error) and apply it to my held‐out test_df.  After transforming the test set, I use the same evaluator to compute the RMSE on that hold‐out data, giving me an unbiased estimate of how well my tuned pipeline will perform in production.

```python
import numpy as np
paramGrid = (ParamGridBuilder()
    .addGrid(lr.regParam,      list(np.arange(0.0, 0.1, 0.01)))
    .addGrid(lr.elasticNetParam, [0.0, 1.0])
    .build()
)

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=RegressionEvaluator(
        labelCol="tip_amount",
        predictionCol="prediction",
        metricName="rmse"
    ),
    numFolds=5,
    parallelism=8              # one task per core node
)

cvModel = crossval.fit(train_df)

bestModel = cvModel.bestModel
preds     = bestModel.transform(test_df)
evaluator = RegressionEvaluator(labelCol="tip_amount",
                                predictionCol="prediction",
                                metricName="rmse")
test_rmse = evaluator.evaluate(preds)

print(f"Best regParam      : {bestModel.stages[-1]._java_obj.getRegParam():.2f}")
print(f"Best elasticNetParam: {bestModel.stages[-1]._java_obj.getElasticNetParam():.2f}")
print(f"Test-set RMSE       : {test_rmse:.2f}")
```

- **Results:**
    - **Best `regParam = 0.00`**  
    A `regParam` of 0 means the model applies **no penalty** on coefficient magnitudes. In other words, the penalty term in the loss function is zero, and the estimator reduces to ordinary least squares.

    - **Best `elasticNetParam = 1.00`**  
    An `elasticNetParam` of 1 would normally select a pure L₁ (lasso) penalty—but since `regParam` is 0, the mixture parameter is moot. The combination effectively yields an **unregularized OLS** model.

    - **Test‐set RMSE = 2.11**  
    On the held‐out 20% of the data, the model’s predictions are off by about \$2.11 on average (root‐mean‐square error).  

### **Feature Importance (Top Coefficients):**

Below are the top 10 features ranked by absolute coefficient magnitude along with their signed effects on **predicted tip amount**:

| Feature            | Coefficient | Interpretation                                                                                       |
|--------------------|------------:|------------------------------------------------------------------------------------------------------|
| **`dist_bin`**     |   +0.6165   | Moving into a higher distance bucket (e.g. from “short” to “mid-range”) increases the predicted tip by \$0.62 on average. Reflects that longer trips tend to yield larger tips. |
| **`pax_ohe`**      |   –0.4696   | Certain passenger-count categories (encoded via one-hot) are associated with lower tips. Larger groups may tip less per person or have different tipping norms. |
| **`is_weekend`**   |   –0.2001   | Trips on Saturdays/Sundays predict \$0.20 less tip on average compared to weekdays.              |
| **`is_airport_trip`** | –0.1792  | Airport-flagged trips predict \$0.18 less tip, controlling for distance bucket.                  |
| **`is_midtown_trip`** | –0.0724  | Midtown trips predict \$0.07 less tip compared to non-Midtown rides.                              |
| **`rush_hour`**    |   +0.0581   | Late-afternoon (4–7 PM) trips predict \$0.06 more tip, likely due to higher demand and congestion. |
| **`corridor_ohe`** |   +0.0165   | Specific high-volume origin–destination corridors contribute small positive adjustments.             |
| **`hour`**         |   +0.0048   | Each additional hour of the day (e.g. 1 PM → 2 PM) adds about \$0.005 to the predicted tip.     |
| **`fare_per_mile`**|   +0.0008   | Higher unit-cost trips (fare divided by distance) add a negligible \$0.001 per dollar-per-mile. |
| **`dist_times_pax`**| –0.0000   | The interaction between distance and passenger count has virtually no effect once other features are accounted for. |

- **Interpretation:**  
    - **Distance buckets dominate**: Nonlinear distance effects (`dist_bin`) are by far the strongest predictor—especially capturing the mid-range generosity bump.  
    - **Group size matters**: The one-hot encoded passenger count (`pax_ohe`) shows that tipping behavior varies non-linearly with party size.  
    - **Temporal flags**: Weekends and airport/Midtown trips tend to tip slightly less, while rush-hour trips tip more.  
    - **Marginal effects**: Fine-grained features like the hour of day or fare‐per-mile have very small per-unit impacts compared to the big categorical and bucketized features.  

Overall, this suggests that modeling tipping behavior benefits most from capturing **nonlinear distance effects**, **passenger group effects**, and **broad categorical flags** (weekend, airport, rush hour), with only marginal gains from continuous interactions like fare-per-mile or exact hour.