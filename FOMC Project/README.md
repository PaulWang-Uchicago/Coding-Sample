# FOMC Sentiment and U.S. Manufacturing New Orders: Text Based Macroeconomic Forecasting
**Zechong (Paul) Wang** \
University of Chicago, MACSS-Econ '26

## Abstract
This project investigates whether the tone of Federal Open Market Committee (FOMC) communications can improve forecasts of monthly U.S. manufacturing new orders. I construct two dictionary-based sentiment indices from FOMC minutes and statements using a curated economic vocabulary and explore their predictive value alongside traditional macroeconomic indicators (e.g., PPI, oil prices, CPI, unemployment). Finding that sentiment scores alone explain little variance in the target, I develop a multi-stage unsupervised pipeline, including TF–IDF clustering, BERTopic, FinBERT tone detection, and PCA, to extract latent textual features from Fed communications. I integrate these with macro series and autoregressive lags into a Random Forest model that achieves high out-of-sample performance (R² = 0.9835, RMSE = 0.042). Despite the rich text-mining framework, results show that realized fundamentals, especially cost pressures and order momentum, remain the dominant drivers of manufacturing behavior, while FOMC tone contributes only marginally. This suggests that while central-bank rhetoric shapes expectations, it exerts limited influence on high-frequency production planning. I conclude that sentiment features may be more useful for structural regime classification than for real-time forecasting of industrial outcomes.

## Workflow Overview

1. **Data Collection**
   - **FOMC Transcripts**  
     Gather raw meeting minutes and policy statements from the Federal Reserve’s website.
   - **Macro Series**  
     Retrieve monthly macroeconomic indicators (e.g., shipments index, ISM new orders) via the FRED API.

2. **Data Preparation**
   - **Text Cleaning & Structuring**  
     Clean raw HTML and pivot so each meeting date maps to:
     - `minute_text`  
     - `statement_text`
   - **Deduplication & Period Conversion**  
     Remove duplicate dates and convert meeting dates to `YYYY-MM` monthly periods.

3. **Exploratory Data Analysis (EDA)**
   - **Sentiment Feature Engineering**  
     Create `score_minute` and `score_statement` using _TF–IDF_ restricted to a custom economic vocabulary.
   - **Temporal Alignment**  
     Interpolate sentiment scores onto a continuous monthly timeline.
   - **Normalization**  
     Scale all predictors to the \[0, 1\] interval.
   - **Descriptive Analysis**  
     Examine feature distributions and compute correlations with the target variable.

4. Model Development

---
## Exploratory Data Analysis (EDA)
### 1. Data Sources and Preparation

  **FOMC Transcripts (Minutes & Statements)**  
  - **Source:**  
    Board of Governors of the Federal Reserve System [website](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)  
  - **Retrieval:**  
    Automated scraping using the script on [GitHub](https://github.com/vtasca/fed-statement-scraping/blob/master/scrape.py) (vtasca, 2020) 
  - **Structure:**  
    Raw HTML is parsed to extract:  
    - **Meeting Date:** Publication date (mapped to the corresponding policy‐meeting month)  
    - **`minute_text`:** Full text of the minutes  
    - **`statement_text`:** Full text of the policy statement

  **New Manufacturing Orders**
  - **Series ID:** `AMTMNO`
  - **Source:** [FRED](https://fred.stlouisfed.org/)
  - **Frequency & Range:** Monthy, January 2000 through Feburary 2025

  **Sticky CPI**
  - **Series ID:** `STICKCPIM157SFRBATL`
  - **Source:** [FRED](https://fred.stlouisfed.org/)
  - **Frequency & Range:** Monthy, January 2000 through Feburary 2025

  **PPI**
  - **Series ID:** `PCUOMFGOMFG`
  - **Source:** [FRED](https://fred.stlouisfed.org/)
  - **Frequency & Range:** Monthy, January 2000 through Feburary 2025

  **Oil Prices**
  - **Series ID:** `WTISPLC`
  - **Source:** [FRED](https://fred.stlouisfed.org/)
  - **Frequency & Range:** Monthy, January 2000 through Feburary 2025

  **Unemployment Rate**
  - **Series ID:** `UNRATE`
  - **Source:** [FRED](https://fred.stlouisfed.org/)
  - **Frequency & Range:** Monthy, January 2000 through Feburary 2025

  **Supply Chain Index**
  - **Series ID:** `gscpi_data`
  - **Source:** [NY Fed](https://www.newyorkfed.org/)
  - **Frequency & Range:** Monthy, January 2000 through Feburary 2025

Raw text and time series are merged on monthly periods to ensure consistent alignment across all features and the target.

### 2. Sentiment Feature Engineering

This section details the creation of two continuous sentiment indices—**`score_minute`** and **`score_statement`**—which quantify the FOMC’s policy tone on a scale from recession (–1) to expansion (+1).

#### 2.1 Vocabulary Curation

- **Categories**  
  - **Expansion** seeds (_e.g._, growth, rebound, hiring)  
  - **Recession** seeds (_e.g._, slump, unemployment, crisis)  
- **Expansion via WordNet**  
  1. Start with ~30 seed words per category.  
  2. For each seed, traverse WordNet synsets and lemmas until ~250 unique, single‐word terms are collected.  
  3. Discard multi‐word entries and non‐alphabetic tokens.  

  ```python
  from nltk.corpus import wordnet as wn

  def expand_terms(seeds, target_size=250):
      terms = set(seeds)
      for seed in seeds:
          for syn in wn.synsets(seed):
              for lemma in syn.lemmas():
                  name = lemma.name().lower()
                  if name.isalpha():
                      terms.add(name)
                  if len(terms) >= target_size:
                      return list(terms)[:target_size]
      return list(terms)

  expansion_terms = expand_terms(expansion_seeds)
  recession_terms  = expand_terms(recession_seeds)

  econ_vocab = {
      "expansion": expansion_terms,
      "recession": recession_terms
  }
  ```

#### 2.2 TF-IDF Matrix Construction
- **Tokenizer:** NLTK's `word_tokenize`
- **Vocabulary:** Combined set of `econ_vocab["expansion"]` and `econ_vocab["recession"]`
- **Transformation:**

  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  import nltk

  all_terms = list(set(expansion_terms) | set(recession_terms))

  tfidf = TfidfVectorizer(
      vocabulary=all_terms,
      tokenizer=nltk.word_tokenize,
      lowercase=True,
      token_pattern=None
  )

  # Minutes corpus to X1
  corpus_minute   = df["minute_text"].astype(str).tolist()
  X1 = tfidf.fit_transform(corpus_minute)

  # Statements corpus to X2
  corpus_statement = df["statement_text"].astype(str).tolist()
  X2 = tfidf.transform(corpus_statement)
  ```

#### 2.3 Score Computation
- **Index Formula**
$\text{score} = \frac{\sum \text{TF–IDF}(\text{expansion}) - \sum \text{TF–IDF}(\text{recession})}
{\sum \text{TF–IDF}(\text{expansion}) + \sum \text{TF–IDF}(\text{recession})}$

- **Implementation**
  ```python
  feat_names = tfidf.get_feature_names_out()
  exp_idx = [i for i,w in enumerate(feat_names) if w in expansion_terms]
  rec_idx = [i for i,w in enumerate(feat_names) if w in recession_terms]

  import numpy as np

  # Sum weights per document
  exp1 = X1[:, exp_idx].sum(axis=1).A1
  rec1 = X1[:, rec_idx].sum(axis=1).A1
  exp2 = X2[:, exp_idx].sum(axis=1).A1
  rec2 = X2[:, rec_idx].sum(axis=1).A1

  # Compute bounded scores, avoiding division by zero
  df["score_minute"]    = np.where((exp1+rec1)>0, (exp1-rec1)/(exp1+rec1), 0.0)
  df["score_statement"] = np.where((exp2+rec2)>0, (exp2-rec2)/(exp2+rec2), 0.0)
  ```

- **Interpretation:** Values near +1 dominant expansionary language; near -1, dominant recessionary language. Zero signals balance. 

### 3. Temporal Alignment & Interpolation

To align sentiment scores with monthly macro series, I perform the following steps:

1. **Convert `Date` to a PeriodIndex** at month-end frequency:
    ```python
    df['Period'] = pd.to_datetime(df['Date'], format='%Y/%m').dt.to_period('M')
    ```

2. **Reindex** to include every month from the first to the last observation:
    ```python
    all_months = pd.period_range(df['Period'].min(), df['Period'].max(), freq='M')
    df_full   = df.set_index('Period').reindex(all_months)

3. **Interpolate** missing sentiment values and fill endpoints: 
    ```python
      # Ensure floats for interpolation
    df_full['score_minute']    = df_full['score_minute'].astype(float)
    df_full['score_statement'] = df_full['score_statement'].astype(float)

    # Linear interpolation + forward/backward fill
    df_full['score_minute']    = df_full['score_minute'].interpolate(method='linear').ffill().bfill()
    df_full['score_statement'] = df_full['score_statement'].interpolate(method='linear').ffill().bfill()
    ```

4. **Restore** the `Date` column in "YYYY/MM" format: 
    ```python
    df_full['Date'] = df_full.index.to_timestamp().strftime('%Y/%m')
    df = df_full.reset_index(drop=True)
    ```

5. **Normalize** the interpolated scores to the [0, 1] interval: 
    ```python
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    df[['score_minute', 'score_statement']] = scaler.fit_transform(
        df[['score_minute', 'score_statement']]
    )
    ```

![sentiment_distribution.png](https://github.com/MACS-33002/macs-33002-2025s-PaulWang-Uchicago/blob/main/Project/figures/sentiment_distribution.png)

### 4. Feature Normalization
- **Min–Max Scaling**  
  Each continuous variable—`score_minute`, `score_statement`, and all economic indicators, is rescaled to the \[0, 1\] interval.

- **Purpose**  
  - Places all predictors on a common scale, preventing higher-magnitude series from dominating model estimation.  
  - Improves numerical stability and convergence in downstream algorithms.  
  - Facilitates straightforward coefficient interpretation and comparison across features.

I chose a PowerTransformer (Box–Cox) on the New Orders series because monthly manufacturing new orders have a long right tail and non‐constant variance. Box–Cox finds an optimal power‐transform lambda to pull in that tail and stabilize variance. 

## Supervised Learning

I leverage the features engineered above (`score_minute`, `score_statement` and five economic indicators) to predict the new manufacturing orders. I fit and compare several tree-based and linear models, then evaluate their out-of-sample performance and feature importances.

### Correlation Analysis  
**Heatmap** of Pearson correlations among all predictors and the target.  

![correlation_heatmap.png](https://github.com/MACS-33002/macs-33002-2025s-PaulWang-Uchicago/blob/main/Project/figures/correlation_heatmap.png)

1. **Minute-level sentiment is moderately predictive**  
   - `score_minute` vs. `shipment_idx`: **+0.36**  
     A positive moderate correlation suggests that spikes in minute-by-minute sentiment scores tend to coincide with higher manufacturing shipments.  
   - By contrast, `score_statement` vs. `shipment_idx` is only **+0.09**, indicating that the broader statement-level sentiment is less informative than the finer-grained minute scores.

2. **Inflation gauges move together**  
   - `CPI` vs. `PPI`: **+0.43**  
     Consumer and producer prices co-move reasonably well, reflecting shared inflationary pressures.  
   - `CPI` vs. `unemployment`: **–0.55**  
     The strong negative correlation is consistent with Phillips-curve dynamics: higher consumer prices often accompany lower unemployment.

3. **Commodity and input-cost links**  
   - `oil_prices` vs. `PPI`: **+0.59**  
     Oil price fluctuations feed directly into producer-price movements, making PPI a good proxy for energy-driven cost changes.

4. **Unemployment dampens shipments**  
   - `unemployment` vs. `shipment_idx`: **–0.47**  
     Rising unemployment is associated with a notable pull-back in manufacturing orders, perhaps due to weaker aggregate demand.

### Data & Train–Test Split  
- **Features (X):**  
  `score_minute`, `score_statement`, `CPI`, `PPI`, `oil_prices`, `unemployment`, `GSCPI`, `shipment_idx`
- **Target (y):**  
  `new_orders_manufacturing` (Box–Cox transformed + scaled)  
- **Split:** 80 % training, 20 % testing (random_state=42)

### Model Training
I apply a suite of models—single decision trees, random forests, gradient‐boosted trees (XGBoost), and polynomial regression—to predict monthly manufacturing new orders from both sentiment and macroeconomic indicators. I chose tree-based methods for their ability to capture non-linear interactions and stabilize predictions via bagging or boosting, and polynomial regression as a transparent, inferential baseline. I expect ensemble models—particularly XGBoost—to deliver the highest out-of-sample accuracy driven mainly by traditional economic series (PPI, oil prices), with sentiment features providing a secondary boost.

| Metric    | Decision Tree | Random Forest | XGBoost  | Polynomial Regression |
|:----------|:--------------|:--------------|:---------|:----------------------|
| **R²**    | 0.916         | 0.977         | 0.9748   | 0.9734                |
| **MAE**   | 0.053         | 0.031         | 0.0311   | 0.0316                |
| **RMSE**  | 0.082         | 0.043         | 0.0449   | 0.0462                |
| **MAPE**  | 12.98%        | 11.12%        | 10.92%   | 14.04%                |

Overall, all models achieve strong predictive performance, but the tree‐based ensembles clearly outperform a single decision tree and the polynomial baseline. The Decision Tree captures basic non‐linearities but leaves substantial error. In contrast, the Random Forest delivers the highest explanatory power ($R^2$ = 0.977) and lowest RMSE (0.043), indicating that bagging and feature subsampling sharply reduce variance. XGBoost closely trails on $R^2$ (0.9748) but attains the lowest MAPE (10.92%), suggesting its boosting algorithm effectively minimizes relative errors on outliers. The Polynomial Regression provides a transparent linear‐plus‐interaction benchmark but yields slightly larger percentage errors, underscoring the value of ensemble methods.

If I had to pick one model to deploy immediately, I’d go with the Random Forest. It delivers the strongest overall fit and lowest RMSE, while still keeping MAPE down. Compared to XGBoost, it’s slightly more robust to hyperparameter settings and offers stable, interpretable feature importances, so I can trust both its accuracy and its insights for decision-making.

Next, I will examine each model’s feature importances to identify which predictors—economic or sentiment—drive my forecasts. This analysis will reveal the relative contribution of variables like PPI, oil prices, and sentiment scores to predictive performance.

#### Decision Tree Regressor
- **Top 5 feature importances**  
  1. **oil_prices** (0.42)  
  2. **PPI** (0.23)  
  3. **CPI** (0.17)  
  4. **shipment_idx** (0.05)  
  5. **unemployment** (0.05)  

#### Random Forest Regressor
- **Top 5 feature importances**  
  1. **PPI** (0.47)  
  2. **oil_prices** (0.24)  
  3. **shipment_idx** (0.08)  
  4. **CPI** (0.08)  
  5. **unemployment** (0.06)  

#### XGBoost Regressor
- **Top 5 feature importances**  
  1. **PPI** (0.81)  
  2. **oil_prices** (0.12)  
  3. **unemployment** (0.04)  
  4. **score_minute** (0.02)  
  5. **shipment_idx** (0.01)  

#### Polynomial Regression Pipeline
- **Top 5 coefficients**  
  1. **PPI** (β ≈ 0.83)  
  2. **oil_prices** (β ≈ 0.29)  
  3. **shipment_idx** (β ≈ 0.16)  
  4. **score_statement** (β ≈ 0.05)  
  5. **score_minute** (β ≈ 0.03)  

#### Feature Importance Interpretation

Across all supervised models, traditional macroeconomic series—especially **PPI**, **oil_prices**, **CPI**, **shipment_idx**, and **unemployment**—consistently rank as the top five predictors. In contrast, my two sentiment scores fall near the bottom of the importance rankings.

- **PPI vs. oil_prices**  
  - In **Random Forest** and **XGBoost**, **PPI** dominates because many trees “vote” on producer‐price splits as the most effective impurity- or error-reduction mechanism.  
  - In a single **Decision Tree**, however, **oil_prices** comes out on top—likely because oil exhibits sharper threshold effects (e.g., sudden price spikes) that a lone tree can exploit early in its splitting hierarchy.  
  - **Polynomial Regression** also assigns the largest coefficient to **PPI** (β ≈ 0.83), reflecting its near-linear, proportional relationship with new orders after scaling.

- **CPI, shipment_idx & unemployment**  
  - **CPI** and **shipment_idx** provide complementary signals of downstream demand and production capacity, respectively; they hover in third or fourth place depending on whether the model emphasizes upstream costs (PPI) or consumer-side pressures (CPI).  
  - **Unemployment** consistently dampens predicted orders and surfaces in the top five for ensemble methods, affirming its tight link to the business cycle.

- **Sentiment scores**  
  - Both `score_minute` and `score_statement` rarely break into the top five and often register near-zero importance. This suggests my dictionary-based TF–IDF sentiment captures only a weak signal relative to hard economic data, and may suffer from noise or insufficient granularity.

Because my simple TF–IDF sentiment indices added little predictive power, I turned to a more comprehensive unsupervised pipeline to extract richer text features:

1. **TF–IDF + K-Means**  
   - Vectorize cleaned minutes with TF–IDF (unigrams + bigrams, custom stopwords).  
   - Fit K-Means on the TF–IDF matrix to assign each meeting to one of eight clusters.  
   - Include these cluster labels (as one-hot dummies) to capture thematic regimes beyond “expansion vs. recession.”

2. **BERTopic + K-Means**  
   - Embed minutes using a Sentence-Transformer model, reduce via UMAP, then apply BERTopic for initial topic discovery.  
   - Override its HDBSCAN step by re-clustering the BERTopic embeddings with K-Means, yielding coherent topic IDs and per-document confidence scores.

4. **PCA on Augmented Feature Set**  
   - Assemble every engineered features into a single matrix.  
   - Apply PCA to distill this high-dimensional space into several orthogonal components that capture the dominant latent patterns.

By augmenting my supervised models with these unsupervised-derived features, I hope to unlock richer, non-linear interactions with macroeconomic series and expect to improve forecasting accuracy.

## Unsupervised Learning

To move beyond my coarse, two‐term TF–IDF sentiment index, I built a multi‐stage pipeline that teases out richer textual patterns. Each step refines the raw meeting minutes into increasingly meaningful features:

- **Discovering Latent Groups with K-Means**  
  After cleaning and TF–IDF vectorization (25 000 unigrams + bigrams, custom stopwords), I ran K-Means for k=2…14 and used the SSE “elbow” and silhouette diagnostics to select k=8. These eight clusters might capture distinct rhetorical regimes—crisis, recovery, tightening, normalization—that my simple sentiment score could never distinguish. Encoding each meeting with a cluster label gives a categorical signal of which policy era it belongs to, directly linking language shifts to economic outcomes.

![elbow_plot.png](https://github.com/MACS-33002/macs-33002-2025s-PaulWang-Uchicago/blob/main/Project/figures/elbow_plot.png)
![silhouette_scores.png](https://github.com/MACS-33002/macs-33002-2025s-PaulWang-Uchicago/blob/main/Project/figures/silhouette_scores.png)

- **Visualizing Structure in UMAP Space**  
  I projected the TF–IDF matrix down to 100 dimensions via SVD, then into 2D with UMAP (cosine metric). The eight K-Means clusters form clear “arms” mirroring historical policy phases, validating that my partition reflects genuine thematic separations. This visualization not only confirms cluster quality but also reveals the chronological flow of Fed discourse, guiding my interpretation of subsequent features.

- **Topic Extraction with BERTopic**  
  To obtain human-interpretable themes, I embedded each document with a Sentence-Transformer model, applied UMAP again, and fit BERTopic to discover latent topics. I then overrode BERTopic’s own clustering with my k=8 labels, producing topic IDs and per-document confidence scores like “Asset Purchases” or “Private Credit” These continuous topic-probability features capture nuanced content differences far beyond raw TF–IDF centroids or polarity bags of words.

![topic_word_scores.png](https://github.com/MACS-33002/macs-33002-2025s-PaulWang-Uchicago/blob/main/Project/figures/topic_word_scores.png)

- **Deep Sentiment with FinBERT**  
  Recognizing that dictionary methods miss context (negation, idioms), I chunked each minute into less than 510-token segments and ran the “yiyanghkust/finbert-tone” pipeline. Averaging the resulting positive, neutral, and negative scores yields a `net_sent` metric that truly understands financial language. This transformer-based sentiment features captures tone shifts—“risk remains” vs. “risks have abated”—that my original two‐term index could not.

![avg_sentiment_FinBERT.png](https://github.com/MACS-33002/macs-33002-2025s-PaulWang-Uchicago/blob/main/Project/figures/avg_sentiment_FinBERT.png)
![sentiment_over_time.png](https://github.com/MACS-33002/macs-33002-2025s-PaulWang-Uchicago/blob/main/Project/figures/sentiment_over_time.png)
![sentiment_new_orders.png](https://github.com/MACS-33002/macs-33002-2025s-PaulWang-Uchicago/blob/main/Project/figures/sentiment_new_orders.png)

- **Dimensionality Reduction with PCA**  
  With dozens of new features (cluster dummies, BERTopic probabilities, FinBERT scores, macro controls), I applied PCA to extract five orthogonal principal components. These components distill the dominant variance across text and economic signals, mitigate multicollinearity, and drastically reduce dimensionality. Feeding these compact, high-signal factors into my supervised models dramatically improves out-of-sample accuracy, demonstrating the power of layered unsupervised feature engineering over simple TF–IDF sentiment.

To harness both real‐time economic signals and the rich thematic structure of FOMC rhetoric, I merge my original macro controls (CPI, PPI, oil prices, unemployment, GSCPI, shipment_idx) and autoregressive lags of the new‐orders series with a diverse set of unsupervised text features:

  ```python
  # List of candidate features
  features = [
      # BERTopic probabilities
      *[c for c in df_model.columns if c.startswith("topic_prob_")],
      "bertopic_km_probability",
      # Cluster dummies
      *[c for c in df_model.columns if c.startswith("clust_")],
      # FinBERT sentiment
      "net_sent", "pos_score", "neg_score", "sent_delta_5", "sent_vol_5",
      # Autoregressive lags & rolling means
      "lag1_orders", "lag2_orders", "roll3_orders",
      # Embedding principal components
      *[c for c in df_model.columns if c.startswith("emb_pc_")],
      # Macroeconomic controls
      "CPI", "PPI", "oil_prices", "unemployment", "GSCPI", "shipment_idx",
      # Satellite data (if available)
      *[c for c in df_model.columns if c.startswith("landsat_")],
      # PCA components
      "pca_1", "pca_2", "pca_3", "pca_4", "pca_5"
  ]
  ```

**Model Performance**  
My Random Forest explains 98.35 % of the variance in held-out new orders ($R^2$ = 0.9835), with an RMSE of 0.042 and MAE of 0.0255. This represents a clear improvement over earlier models that relied solely on supervised TF–IDF sentiment scores, which contributed almost nothing to predictive power.

![pred.png](https://github.com/MACS-33002/macs-33002-2025s-PaulWang-Uchicago/blob/main/Project/figures/pred.png)

**Feature Importances**  
- The first principal component (`pca_1`), which blends my unsupervised text factors with core macro signals, accounts for 46.7 % of the model’s split decisions—evidence that thematic regimes and topic probabilities now carry genuine predictive weight.  
- The one‐meeting lag (`lag1_orders`, 35.3%) and two‐meeting lag (`lag2_orders`, 4.0%), along with the three-meeting rolling mean (`roll3_orders`, 3.2%), confirm strong autoregressive momentum in manufacturing orders.  
- Among pure macro variables, the Producer Price Index (8.1%) and oil prices (1.1%) remain the next most important predictors.  
- Cluster, topic, and deep-sentiment features each register below 0.3% individually, but collectively they account for several percentage points of importance—an encouraging advance from the near-zero importances seen in the supervised‐only phase.

**Secondary Predictors**  
- **`oil_prices` (0.011)** continues to matter, albeit less so.  
- Unsupervised text signals like **`bertopic_km_probability` (0.0022)** and cluster dummies, as well as classical macro controls (GSCPI, shipment_idx, CPI, unemployment), each contribute only marginally (<0.002).  
- **FinBERT sentiment** features (`net_sent`, `pos_score`, `neg_score`, `sent_vol_5`, `sent_delta_5`) all lie near zero importance, reinforcing that my text-derived signals are still dwarfed by momentum and price indices.

## Statistical Inference & Error Analysis
### Assessing the Value of Sentiment Features
Random forests do not yield p-values in the classical sense, but I can still gauge the statistical importance of my FinBERT and clustering–derived sentiment features through a combination of permutation‐based tests and model comparison:

#### Permutation Importance Test
  - I randomly permute each sentiment feature (e.g. `net_sent`, `sent_delta_5`) in the test set—breaking its relationship with the target—while keeping all other predictors fixed.  
  - The average **drop in R²** across many permutations serves as an empirical “significance” measure: features whose shuffling causes a meaningful decline in forecast accuracy are judged important.  
  - In my case, permuting `net_sent` and `pos_score` causes negligible mean_drop and std_drop of less than 0.001, suggesting their marginal contribution is statistically indistinguishable from noise.

    | Feature        | Mean ΔR² Drop | Std ΔR² Drop |
    |:---------------|--------------:|-------------:|
    | **sent_delta_5** |      0.000092 |     0.000107 |
    | **neg_score**    |      0.000077 |     0.000073 |
    | **sent_vol_5**   |      0.000041 |     0.000223 |
    | **net_sent**     |      0.000036 |     0.000064 |
    | **pos_score**    |      0.000004 |     0.000131 |


#### Nested Model Comparison
  - I fit two Random Forests on the same train–test split:  
     1. **Full model** with all features, including sentiment.  
     2. **Reduced model** excluding all sentiment‐derived columns.  
   - Comparing out‐of‐sample RMSE and R² via a paired bootstrap (resampling test‐set predictions) allows me to test whether the difference in error distributions is statistically significant.  
   - The paired‐bootstrap test shows that dropping all sentiment features (net_sent, pos_score, neg_score, sent_delta_5, sent_vol_5) improves predictive accuracy:
	    - Full model RMSE = 0.042
	    - Reduced model RMSE = 0.039
	    - 95 % CI for $\Delta$ RMSE (reduced – full) = [–0.0023, –0.0001]

![full_reduced_models.png](https://github.com/MACS-33002/macs-33002-2025s-PaulWang-Uchicago/blob/main/Project/figures/full_reduced_models.png)

Because the entire confidence interval lies below zero, the reduction in RMSE is statistically significant at the 5 % level. In other words, including those sentiment metrics actually worsens out‐of‐sample performance by about 0.001 RMSE units.

#### Residuals vs. Sentiment

![resid_sentiment.png](https://github.com/MACS-33002/macs-33002-2025s-PaulWang-Uchicago/blob/main/Project/figures/resid_sentiment.png)

Across all five plots, the regression lines are essentially flat and the shaded confidence bands are wide, indicating no meaningful slope. In other words, neither raw net sentiment, its five‐meeting change or volatility, nor the FinBERT positive/negative scores systematically explain the remaining forecast errors. Any points with large residuals (e.g. November 2010 or February 2020) occur at extreme macro shocks rather than at particular values of sentiment. This confirms quantitatively and visually that once I control for momentum, cost pressures, clusters, topics, and principal components, my model’s misspecifications are orthogonal to Fed‐minute tone—there is no residual bias tied to high or low sentiment. Thus, sentiment features, while conceptually appealing, do not underlie the patterns in my forecast errors.

### Error Analysis
To ensure my model’s robustness and to diagnose any systematic biases, I examine its residuals and large‐error cases:

#### Residual Distribution

![resid_dist.png](https://github.com/MACS-33002/macs-33002-2025s-PaulWang-Uchicago/blob/main/Project/figures/resid_dist.png)

The violin‐and‐boxplot visualization reveals that the Random Forest’s forecast errors are both small and well behaved. The median residual lies almost exactly at zero and the mean is similarly centered, indicating no systematic over‐ or under‐prediction. Roughly half of all errors fall within a narrow ±0.02 window (the interquartile range), and even the most extreme misses rarely exceed ±0.06, demonstrating that outliers are scarce. The roughly symmetric “violin” shape—with only a slight positive skew—confirms homoscedasticity: error variance does not blow up at particular levels of new orders. Together, these patterns underscore the model’s stability and unbiasedness across tranquil and turbulent economic regimes alike.

### Time‐Series Error Trends 

![time_series_resid.png](https://github.com/MACS-33002/macs-33002-2025s-PaulWang-Uchicago/blob/main/Project/figures/time_series_resid.png)

The chronologically-ordered residual plot shows that errors remain small and centered around zero for most of the sample, but it also reveals a few systematic biases during major shocks. Immediately after the 2008 crisis, the five-meeting rolling mean climbs to around +0.05, indicating the model was consistently under-predicting new orders as the economy unexpectedly rebounded. A similar, if less pronounced, positive bias appears during the mid-2010s expansion. Conversely, the rolling mean dips slightly below zero around the 2020 COVID shock—reflecting a brief tendency to over-predict manufacturing activity when orders plunged. Outside these episodes, the smoothed residuals hover within ±0.02, confirming that the Random Forest maintains unbiased, homoscedastic performance across both tranquil and turbulent policy regimes. Overall, this time-series view underscores the model’s robustness while highlighting its small, predictable mis-calibrations at structural turning points.

### Top 10 Largest Errors 

| Date    |   Actual | Predicted |  Residual |  AbsError |
|:--------|---------:|----------:|----------:|----------:|
| 2010/11 |  0.440111|   0.319138|   0.120973|   0.120973|
| 2011/12 |  0.618261|   0.559392|   0.058869|   0.058869|
| 2010/05 |  0.361865|   0.312095|   0.049770|   0.049770|
| 2020/02 |  0.562171|   0.516097|   0.046074|   0.046074|
| 2008/03 |  0.602944|   0.562331|   0.040614|   0.040614|
| 2001/04 |  0.052621|   0.091009|  −0.038388|   0.038388|
| 2021/12 |  0.850436|   0.885930|  −0.035494|   0.035494|
| 2017/09 |  0.560016|   0.528485|   0.031531|   0.031531|
| 2017/02 |  0.530958|   0.500301|   0.030657|   0.030657|
| 2022/07 |  0.959282|   0.929598|   0.029685|   0.029685|

The largest forecast errors all occur at the sharpest turning points in manufacturing activity—periods when real‐world orders either snapped back or plunged more abruptly than my model anticipated. For example, the single biggest miss (+0.12) came in November 2010, as orders rebounded strongly following the Global Financial Crisis, followed by further under‐predictions in mid‐2010 and late‐2011. A similar pattern appears in February 2020, on the eve of the COVID‐19 shock, when my model again underestimated the sudden swing. By contrast, the model tends to overshoot at deep troughs—most notably in April 2001 after the dot-com downturn and in December 2021—reflecting a modestly optimistic bias when orders reach their lowest levels. Importantly, even these “worst‐case” residuals remain small relative to the normalized [0, 1] scale, confirming the model’s strong overall performance. Nonetheless, these inflection‐point errors highlight an inherent challenge: rapid regime shifts and extreme shocks, which unfold faster than my lagged macro inputs and policy‐text features can fully capture, will always be harder to predict perfectly.

## Reflection on Feature Selection

### Reflection on Sentiment Features

My feature-engineering journey began with the intuitive appeal of TF–IDF-based sentiment scores (`score_minute`, `score_statement`), which promised to capture the Fed’s oscillation between optimism and caution. However, these two indices alone explained almost none of the variance in manufacturing new orders: correlation and feature-importance analyses showed that upstream cost pressures (PPI, oil prices) and order momentum (lagged new orders) overwhelmingly dominated the predictive landscape.

Rather than discard text entirely, I enriched my feature set with unsupervised methods:

- **K-Means clustering** on the TF–IDF matrix revealed eight thematic regimes—“crisis,” “tightening,” “recovery,” etc.—that simple polarity scores could not distinguish.  
- **BERTopic probabilities** used contextual embeddings to uncover interpretable topics (e.g., *Quantitative Easing*, *Credit Conditions*), assigning each meeting a continuous confidence weight.  
- **FinBERT sentiment** introduced a transformer-based tone detector, sensitive to negations and financial jargon.  
- **PCA** distilled dozens of macro and text-derived signals into a handful of orthogonal components capturing the dominant variance.

When these unsupervised features joined my core macro controls in a Random Forest, the first principal component alone accounted for nearly **47%** of split decisions—clear evidence that thematic regimes and latent topics drive far more predictive power than off-the-shelf polarity measures.

Yet even this richer text representation could not elevate sentiment into a primary driver. Detailed importance rankings, permutation tests, and paired-bootstrap RMSE comparisons showed that including any of my sentiment metrics (`net_sent`, `pos_score`, `neg_score`, `sent_delta_5`, `sent_vol_5`) **worsened** out-of-sample accuracy (95% CI for ΔRMSE [–0.0023, –0.0001]).

#### Why sentiment underperformed:

1. **Timing and lag**  
   FOMC minutes are released three weeks after each meeting—too late to influence the monthly orders they aim to predict.  
2. **Consensus-driven language**  
   Policy texts are carefully hedged and consensus-oriented, muting the variability that sentiment scoring requires to detect short-term shifts.  
3. **Low signal-to-noise ratio**  
   Even FinBERT, trained on financial corpora, struggles to disentangle boilerplate policy jargon from genuine tonal changes.

From an economic perspective, manufacturing decisions follow the **accelerator principle**—firms adjust rapidly to recent output—and **cost-push dynamics**, where input prices immediately affect margins and inventory. By contrast, central-bank rhetoric primarily shapes **medium-term expectations** rather than high-frequency production plans.

Looking forward, sentiment is best viewed as a **contextual enrichment**—useful for regime detection or structural-break analysis—rather than a standalone forecasting tool. Any future “rescue” of textual signals should focus on deeper, unlabeled features (e.g., evolving embedding trajectories, real-time news flows) that align more closely with firms’ decision horizons.

### Discussion on Macro Data Selection

Although my unsupervised text features played a supporting role, the choice of macroeconomic controls was **crucial** for forecasting accuracy. Each series captures a distinct dimension of the manufacturing environment:

1. **New Manufacturing Orders (`AMTMNO`)**  
   - **Role:** Dependent variable
   - **Why:** Direct outcome of firms’ production planning and inventory investment—reflects real-time demand.

2. **Producer Price Index (`PPI`)**  
   - **Role:** Upstream cost pressure (selling prices received by producers)  
   - **Why:** Under cost-push inflation, rising PPI tightens margins and influences reorder timing; it also serves as a forward-looking indicator of producer confidence.

3. **Oil Prices (`WTISPLC`)**  
   - **Role:** Key commodity input cost  
   - **Why:** Energy costs feed into transport and production; high oil prices often coincide with PPI spikes and add volatility to inventory management.

4. **Sticky Consumer Price Index (`STICKCPIM157SFRBATL`)**  
   - **Role:** Downstream demand conditions (core inflation measure)  
   - **Why:** Reflects the persistence of consumer-side price pressures and affects purchasing power.

5. **Unemployment Rate (`UNRATE`)**  
   - **Role:** Labor-market slack  
   - **Why:** A leading indicator of aggregate demand; rising unemployment signals lower production and ordering activity.

6. **Global Supply Chain Pressure Index (`GSCPI`)**  
   - **Role:** External cost and disruption measure (NY Fed index)  
   - **Why:** Captures logistics shocks and intermediate goods bottlenecks that impact production schedules and inventories.

7. **Shipments Index (`FRGSHPUSM649NCIS`)**  
   - **Role:** Capacity utilization proxy  
   - **Why:** Strongly co-moves with new orders; reflects firms’ tendency to match production with recent shipment levels.

By combining these six macro series with autoregressive lags of new orders and unsupervised text features, I constructed a **comprehensive feature set** that spans demand momentum, cost pressures, consumer conditions, and global logistics dynamics. In practice, **PPI** and **oil prices** emerged as the most important predictors, followed by order momentum lags—validating the central role of **cost-push** and **accelerator mechanisms** in high-frequency manufacturing forecasting.

## Limitations
While this project makes meaningful strides in fusing unsupervised text analysis with economic forecasting, several limitations warrant consideration. First, all macroeconomic indicators used are final-released, backward-looking series. In reality, economic agents operate under conditions of uncertainty, responding to real-time data releases that may later be revised. Thus, my model may overstate predictive power compared to what would be feasible in a true forecasting context using real-time vintages.

Second, the textual corpus is limited to official FOMC meeting minutes and statements, which are only two of several communication channels used by policymakers. High-frequency signals embedded in speeches, testimonies, or unscheduled announcements are excluded, potentially omitting valuable forward guidance. Moreover, I treat all documents as homogeneous, ignoring possible heterogeneity in tone between Chair-led sections and committee-wide commentary.

Third, the current feature engineering framework assumes temporal homogeneity in the relationship between sentiment, macro fundamentals, and manufacturing behavior. That is, it presumes the functional mapping from text to economic behavior remains constant across regimes. In reality, the impact of Fed communication may vary across time—e.g., during crises, markets may place more weight on tone than during stable periods.

Finally, while dimensionality reduction techniques like PCA effectively summarize variation across dozens of features, they sacrifice direct interpretability, potentially obscuring nuanced mechanisms linking sentiment, macro signals, and production decisions. More interpretable alternatives (e.g., partial dependence plots, SHAP) could complement these latent components in future work.

## Policy Implications
The empirical results of this project offer nuanced implications for the role of central-bank communication in shaping real economic activity. Despite the intuitive appeal of using FOMC tone to forecast manufacturing orders, my analysis shows that sentiment-based features—whether hand-crafted or learned via transformers—do not significantly enhance predictive accuracy. Once traditional macro fundamentals and lagged momentum are accounted for, sentiment contributes little marginal signal.

This finding aligns with the broader monetary-policy literature: central-bank communication serves a critical function in anchoring medium-term expectations, but its immediate influence on production and inventory behavior appears limited. Manufacturing firms—unlike financial market participants—may be less reactive to rhetorical signals and more attuned to realized cost pressures (e.g., PPI, oil prices) and internal order flows.

For policymakers, this suggests that communication strategies are most effective when targeting expectation formation and financial-market stability, rather than when attempting to guide real-sector responses at high frequency. Efforts to enhance communication transparency, clarity, and timeliness remain essential—but should be understood as complements to, rather than substitutes for, traditional monetary instruments. Moreover, in times of rapid economic transition, such as the post-COVID rebound or inflationary spikes, hard data still appears to be the primary driver of real activity.

In short, while central-bank tone shapes perceptions, it is realized fundamentals—not rhetoric—that move production. Future research could explore whether communication has stronger predictive power in financial channels (e.g., equity or credit markets), or whether alternative text sources—such as business outlook surveys or firm-level sentiment—hold greater promise for forecasting real economic outcomes.

## Conclusion
This project set out to evaluate the predictive value of FOMC sentiment for short-term manufacturing dynamics. While initial efforts using dictionary-based polarity scores showed limited explanatory power, unsupervised methods—including topic modeling, cluster analysis, and transformer-based tone extraction—revealed richer linguistic structure embedded in monetary policy discourse. These text-derived features, when fused with macroeconomic controls in ensemble models, modestly improved performance but were ultimately overshadowed by classical predictors: PPI, oil prices, shipment momentum, and unemployment.

The findings highlight a clear hierarchy of signals in forecasting monthly manufacturing new orders. First-order variation is driven by cost-push and momentum forces, which capture the mechanics of firms adjusting production in response to recent activity and input costs. Second-order refinements come from autoregressive smoothing and dimensionality reduction over joint economic-textual spaces. Sentiment, though conceptually appealing and methodologically sophisticated, appears better suited for medium-term regime detection than for high-frequency prediction.

The project demonstrates how a combination of machine learning, text mining, and economic theory can enrich my understanding of production behavior. But it also offers a caution: not all signals translate into near-term action, and tone—even when carefully measured—may lag behind or simply echo the harder realities already reflected in prices and output. Going forward, the path to better forecasting may lie not in more sentiment, but in integrating alternative high-frequency text sources, real-time data releases, and decision-timing heterogeneity across firms.
