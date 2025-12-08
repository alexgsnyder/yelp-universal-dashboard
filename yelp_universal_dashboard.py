#!/usr/bin/env python
# coding: utf-8

# # Yelp Universal Business Dashboard
# 
# **Course:** MSBA 503 – Analytics Programming II  
# **Team:** Alex, Eddie, Lily  
# **Project:** Yelp-Based Universal Business Analytics Dashboard  
# 
# This notebook implements a modular, scalable `yelp_dashboard()` master function that:
# 
# - Accepts user/industry filters (Hair, Mexican Food, etc.)
# - Loads Yelp review data and Business Aggregated data
# - Performs descriptive, diagnostic, predictive, and prescriptive analysis
# - Launches an interactive Streamlit dashboard for business users
# 
# > **Design constraints:**  
# > - Do **not** rename any existing columns or files.  
# > - Drop overly granular `_pct` columns (e.g., `adv_pct`, `noun_pct`, `adj_pct`, `verb_pct`) **only during data loading**, not in the source files.  
# > - Final code should end with a callable `yelp_dashboard()` master function.

# In[33]:


#2. Imports & File Paths

# Core libraries
import polars as pl
import pandas as pd
import numpy as np

# Dashboard / visualization
import streamlit as st
import altair as alt

# Modeling & time series
from statsmodels.tsa.arima.model import ARIMA

# Utilities
import random
import time
from datetime import datetime, timedelta
import re
import warnings
import os

# Suppress common warnings during development
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# File paths (match your actual folder)
# ---------------------------------------------------------------------
SAMPLE_PATH_HAIR = "hair_sample_15k.parquet"
SAMPLE_PATH_MEXICAN = "chipotle_sample_15k.parquet"
BUSINESS_AGGREGATED_PATH = "business_aggregated_sample.csv"

# NOTE:
# - Do NOT rename or move these files in the final project.
# - The actual .parquet and .csv files should already be in the /Datasets folder
#   relative to this notebook/script.


# ## 3. Helper & Utility Functions (Scaffold)
# Below we define reusable components for:
# 
# - Data loading & preprocessing  
# - Filtering based on user selections  
# - Shared utilities for analysis functions  
# 
# We will gradually fill these functions in as we refactor Lily's prototype code.

# In[34]:


#3. Helper & Utility Functions (Scaffold)

def load_and_preprocess_data(industry_choice: str):
    """
    Load Yelp review data (Hair or Mexican Food) and Business Aggregated data.

    - Uses Polars for performance.
    - Drops overly granular *_pct columns (adv_pct, noun_pct, adj_pct, verb_pct)
      ONLY in-memory, not from the source files.
    - Joins reviews with business-level aggregated features on `business_id`.
    
    Returns
    -------
    df_joined : pl.DataFrame
        Joined review + business features for the selected industry.
    business_lookup : pl.DataFrame
        Unique (business_id, name, city, state, ...) rows for UI filtering.
    """
    st.info(f"Loading Yelp Data for {industry_choice}...")

    # 1. Choose the correct parquet path based on industry
    path = SAMPLE_PATH_MEXICAN if industry_choice == "Mexican Food" else SAMPLE_PATH_HAIR

    if not os.path.exists(path):
        st.error(f"Review Parquet file not found at {path}.")
        return None, None

    # 2. Load review data (Parquet)
    reviews_df = pl.read_parquet(path)

    # 3. Load business aggregated data (CSV)
    st.info("Loading Business Aggregated Data...")
    try:
        business_df = pl.read_csv(BUSINESS_AGGREGATED_PATH)
    except pl.ComputeError:
        st.error(f"Business Aggregated CSV not found at {BUSINESS_AGGREGATED_PATH}.")
        return None, None

    # 4. Drop overly granular *_pct columns if present
    cols_to_drop = ['adv_pct', 'noun_pct', 'adj_pct', 'verb_pct']
    existing_to_drop = [c for c in cols_to_drop if c in reviews_df.columns]
    if existing_to_drop:
        reviews_df = reviews_df.drop(existing_to_drop)

    # 5. Join on business_id
    st.info("Now joining industry data on 'business_id'...")
    df_joined = reviews_df.join(business_df, on='business_id', how='left')

    # 6. Mock SEC / financial statistics (placeholder logic)
    st.info("Finding Financial Statistics (MOCK)...")
    if 'avg_star' in df_joined.columns and 'stars' in df_joined.columns:
        df_joined = df_joined.with_columns(
            (
                pl.col('avg_star') * 10000 +
                pl.col('stars') * 5000 +
                pl.lit(random.randint(50_000, 200_000))
            ).alias('yearly_revenue')
        )
    else:
        # If these columns don't exist, still create a placeholder yearly_revenue
        df_joined = df_joined.with_columns(
            pl.lit(random.randint(50_000, 200_000)).alias('yearly_revenue')
        )

        # 7. Build a lookup table for business-level info
    base_cols = ['business_id', 'name', 'city', 'state']

    # Try to bring through business-level star metrics from the aggregated data
    star_cols = [
        'weighted_star_avg',
        'business_weighted_star_avg',
        'business_star_avg',
        'stars'
    ]
    lookup_cols = [c for c in base_cols + star_cols if c in df_joined.columns]

    business_lookup = df_joined.select(lookup_cols).unique()

    return df_joined, business_lookup


def apply_user_filters(df_full: pl.DataFrame,
                       business_lookup: pl.DataFrame,
                       business_id_filter=None,
                       business_name=None,
                       state_filter=None,
                       city_filter=None):
    """
    Apply user-specified filters in the correct priority order:

    Priority
    --------
    1. Exact business_id (highest priority)
    2. Business name
    3. City (within a state)
    4. State
    5. Otherwise: industry-wide

    Returns
    -------
    df_analysis : pl.DataFrame
        Filtered dataset ready for analysis.
    subset_label : str
        Human-readable label describing the subset (for dashboard display).
    """
    df_filtered = df_full.lazy()
    subset_label = "Industry-wide"

    # 1. Business ID filter
    if business_id_filter:
        df_filtered = df_filtered.filter(pl.col('business_id') == business_id_filter)
        subset_label = f"Business ID: {business_id_filter}"

    # 2. Business name filter
    elif business_name and business_name != "Industry-Level":
        if 'name' in business_lookup.columns:
            biz_ids = (
                business_lookup
                .filter(pl.col('name') == business_name)
                .select('business_id')
                .to_series()
                .to_list()
            )
            df_filtered = df_filtered.filter(pl.col('business_id').is_in(biz_ids))
            subset_label = f"Business: {business_name}"

    # 3. City filter
    elif city_filter and 'city' in df_full.columns:
        df_filtered = df_filtered.filter(pl.col('city') == city_filter)
        if state_filter and 'state' in df_full.columns:
            df_filtered = df_filtered.filter(pl.col('state') == state_filter)
            subset_label = f"City: {city_filter}, {state_filter}"
        else:
            subset_label = f"City: {city_filter}"

    # 4. State filter
    elif state_filter and 'state' in df_full.columns:
        df_filtered = df_filtered.filter(pl.col('state') == state_filter)
        subset_label = f"State: {state_filter}"

    # 5. Otherwise: industry-wide
    else:
        subset_label = "Industry-wide"

    df_analysis = df_filtered.collect()
    return df_analysis, subset_label


# ## Descriptive Analysis

# In[35]:


def descriptive_analysis(df: pl.DataFrame, business_lookup: pl.DataFrame):
    """
    Descriptive analysis for a filtered Yelp subset.

    Returns
    -------
    line_chart : alt.Chart
        Actual vs time-decay–weighted average star ratings over time.
    pie_chart : alt.Chart or None
        Emotion distribution (if dominant_emotion or similar is available).
    top_5_best : pl.DataFrame
        Top 5 businesses by avg star rating (with review_count).
    top_5_worst : pl.DataFrame
        Bottom 5 businesses by avg star rating (with review_count).
    """
    import altair as alt
    import pandas as pd

    # -------------------------------------------------
    # 0. Basic guard
    # -------------------------------------------------
    if df is None or len(df) == 0:
        return None, None, pl.DataFrame(), pl.DataFrame()

    # -------------------------------------------------
    # 1. Prepare time series (truncate to month)
    # -------------------------------------------------
    try:
        df_ts = df.with_columns(
            pl.col("review_date")
              .str.strptime(pl.Date, strict=False)
              .cast(pl.Datetime)
              .dt.truncate("1mo")
        )
    except Exception:
        df_ts = df.with_columns(
            pl.col("review_date").cast(pl.Datetime).dt.truncate("1mo")
        )

    # -------------------------------------------------
    # 2. Detect weighted column & scale if needed
    # -------------------------------------------------
    weight_col = None

    # First, look for explicit "weight"/"tdw" star columns
    weight_candidates = [
        c for c in df_ts.columns
        if (
            ("weight" in c.lower() and "star" in c.lower())
            or ("tdw" in c.lower() and "star" in c.lower())
        )
    ]

    # If we didn't find any, fall back to:
    # "any star column that is NOT just 'stars' or a simple aggregate"
    if not weight_candidates:
        star_cols = [c for c in df_ts.columns if "star" in c.lower()]
        weight_candidates = [
            c for c in star_cols
            if c.lower() not in ["stars", "business_star_avg", "avg_stars"]
        ]

    if weight_candidates:
        weight_col = weight_candidates[0]

    # Optional: print once to confirm which column we picked
    # print("Using weighted column:", weight_col)

    # If we don't find a weighted column, we'll only plot actual stars
    scale_factor = 1.0
    if weight_col is not None:
        max_weight = df_ts.select(pl.col(weight_col).max()).item()
        if max_weight is not None and max_weight <= 1.5:
            scale_factor = 5.0  # bring 0–1 up to roughly 0–5

    # -------------------------------------------------
    # 3. Aggregate monthly means
    # -------------------------------------------------
    agg_exprs = [pl.col("stars").mean().alias("Actual_Stars")]
    if weight_col is not None:
        agg_exprs.append(
            (pl.col(weight_col) * scale_factor)
            .mean()
            .alias("Weighted_Stars")
        )

    df_ts_agg = (
        df_ts
        .group_by("review_date")
        .agg(agg_exprs)
        .sort("review_date")
    )

    # Convert to long format for Altair
    pdf = df_ts_agg.to_pandas()

    # Melt only the columns that actually exist
    value_vars = [c for c in ["Actual_Stars", "Weighted_Stars"] if c in pdf.columns]

    ts_long = pdf.melt(
        id_vars="review_date",
        value_vars=value_vars,
        var_name="Rating_Type",
        value_name="Average_Star_Rating",
    )

    # -------------------------------------------------
    # 4. Build the line chart
    # -------------------------------------------------
    line_chart = (
        alt.Chart(ts_long)
        .mark_line()
        .encode(
            x=alt.X("review_date:T", title="Date"),
            y=alt.Y(
                "Average_Star_Rating:Q",
                title="Average Star Rating",
                scale=alt.Scale(domain=[0, 5])
            ),
            color=alt.Color(
                "Rating_Type:N",
                title="Rating Type",
                sort=["Actual_Stars", "Weighted_Stars"]
            ),
            tooltip=[
                alt.Tooltip("review_date:T", title="Date"),
                alt.Tooltip("Rating_Type:N", title="Type"),
                alt.Tooltip("Average_Star_Rating:Q", title="Avg Stars", format=".2f"),
            ],
        )
        .properties(
            title="Star Ratings (Actual vs Time-Decay Weighted) Over Time",
            width=700,
            height=400,
        )
    )

    # -------------------------------------------------
    # 5. Emotion pie chart (if available)
    # -------------------------------------------------
    pie_chart = None
    emo_col = None
    for c in ["dominant_emotion", "primary_emotion_mode", "dominant_emotion_right"]:
        if c in df.columns:
            emo_col = c
            break

    if emo_col is not None:
        emo_counts = (
            df
            .group_by(emo_col)
            .agg(pl.count().alias("review_count"))
            .sort("review_count", descending=True)
            .to_pandas()
        )

        pie_chart = (
            alt.Chart(emo_counts)
            .mark_arc()
            .encode(
                theta=alt.Theta("review_count:Q", title="Number of Reviews"),
                color=alt.Color(f"{emo_col}:N", title="Emotion"),
                tooltip=[emo_col, "review_count"],
            )
            .properties(
                title="Distribution of Dominant Emotions",
                width=400,
                height=400,
            )
        )

    # -------------------------------------------------
    # 6. Top 5 Best / Worst businesses by avg stars
    # -------------------------------------------------
    if "business_id" in df.columns and "stars" in df.columns:
        biz_agg = (
            df
            .group_by("business_id")
            .agg([
                pl.col("stars").mean().alias("avg_stars"),
                pl.count().alias("review_count"),
            ])
        )

        # attach business name / location
        if "business_id" in business_lookup.columns:
            biz_agg = biz_agg.join(business_lookup, on="business_id", how="left")

        biz_pdf = biz_agg.to_pandas()

        top_5_best = (
            biz_pdf.sort_values(["avg_stars", "review_count"], ascending=[False, False])
                  .head(5)
        )
        top_5_worst = (
            biz_pdf.sort_values(["avg_stars", "review_count"], ascending=[True, False])
                  .head(5)
        )

        # Convert back to Polars for consistency with your earlier return type
        top_5_best = pl.from_pandas(top_5_best)
        top_5_worst = pl.from_pandas(top_5_worst)
    else:
        top_5_best = pl.DataFrame()
        top_5_worst = pl.DataFrame()

    return line_chart, pie_chart, top_5_best, top_5_worst


# ## Predictive Analysis

# In[36]:


def predictive_analysis(df: pl.DataFrame, periods: int = 6):
    """
    Predict future average star ratings using ARIMA.

    Parameters
    ----------
    df : pl.DataFrame
        Filtered review dataset.
    periods : int
        Number of months to forecast.

    Returns
    -------
    forecast_chart : alt.Chart or None
        Chart with historical + forecasted ratings.
    predicted_star : float
        Star prediction for the final forecasted month.
    """

    # ---------------------------
    # 1. Ensure review_date exists
    # ---------------------------
    if "review_date" not in df.columns or "stars" not in df.columns:
        raise ValueError("DataFrame must contain 'review_date' and 'stars' columns.")

    # ---------------------------
    # 2. Parse review_date → month
    # ---------------------------
    try:
        df_ts = df.with_columns(
            pl.col("review_date")
            .str.strptime(pl.Date, strict=False)
            .cast(pl.Datetime)
            .dt.truncate("1mo")
        )
    except Exception:
        df_ts = df.with_columns(
            pl.col("review_date")
            .cast(pl.Datetime)
            .dt.truncate("1mo")
        )

    # ---------------------------
    # 3. Build monthly star series
    # ---------------------------
    ts_df = (
        df_ts
        .group_by("review_date")
        .agg(pl.col("stars").mean().alias("mean_star"))
        .sort("review_date")
        .to_pandas()
        .set_index("review_date")
    )

    # Resample monthly (ensure continuous)
    ts_df = ts_df.resample("M").ffill()

    # If dataset too small → fallback
    if len(ts_df) < 12:
        last_val = float(ts_df["mean_star"].iloc[-1])
        return None, last_val

    # ---------------------------
    # 4. Fit ARIMA Model
    # ---------------------------
    try:
        model = ARIMA(ts_df["mean_star"], order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast next N months
        forecast = model_fit.predict(
            start=len(ts_df),
            end=len(ts_df) + periods - 1
        )

        # Build forecast index
        forecast_index = [
            ts_df.index[-1] + pd.DateOffset(months=i+1)
            for i in range(periods)
        ]

        forecast_df = pd.DataFrame({
            "review_date": forecast_index,
            "mean_star": forecast.values,
            "type": "Forecast"
        })

        # Historical df
        hist_df = ts_df.reset_index()
        hist_df["type"] = "Historical"

        # Combine for plotting
        combined_df = pd.concat([hist_df, forecast_df], ignore_index=True)

                # ---------------------------
        # 5. Build Altair Forecast Chart
        # ---------------------------
        
        # Separate historical and forecast for clean layering
        hist_pd = hist_df.copy()
        fc_pd = forecast_df.copy()

        chart_hist = (
            alt.Chart(hist_pd)
            .mark_line(color="#f28e2b", strokeWidth=2)  # Orange
            .encode(
                x=alt.X("review_date:T", title="Date"),
                y=alt.Y("mean_star:Q", title="Average Star Rating",
                        scale=alt.Scale(domain=[0, 5])),
                tooltip=[
                    alt.Tooltip("review_date:T", title="Date"),
                    alt.Tooltip("mean_star:Q", title="Avg Stars", format=".2f"),
                ],
            )
        )

        chart_fc = (
            alt.Chart(fc_pd)
            .mark_line(color="#4e79a7", strokeWidth=2)  # Blue
            .encode(
                x="review_date:T",
                y="mean_star:Q",
                tooltip=[
                    alt.Tooltip("review_date:T", title="Date"),
                    alt.Tooltip("mean_star:Q", title="Forecast", format=".2f"),
                ],
            )
        )

        forecast_chart = (
            (chart_hist + chart_fc)
            .properties(
                title=f"Star Rating Forecast ({periods} Months)",
                width=700,
                height=400,
            )
        )

        predicted_star = float(forecast.values[-1])

        return forecast_chart, predicted_star

    except Exception as e:
        print("ARIMA failed:", e)
        last_val = float(ts_df["mean_star"].iloc[-1])
        return None, last_val


# ## Diagnostic Analysis

# In[37]:


def diagnostic_analysis(
    df: pl.DataFrame,
    business_lookup: pl.DataFrame,
    top_n: int = 20
):
    """
    Diagnostic analysis (Eddie's section).

    Identifies:
    - Pain points using keyword frequency in low-star reviews.
    - Emotion intensity summaries (anger, fear, joy, etc.) when available.
    - Linguistic correlations with star ratings.
    - Business-level “at risk” flags based on recent review declines.

    Returns
    -------
    dict
        {
          'pain_point_chart': alt.Chart or None,
          'pain_points_df': pandas.DataFrame,
          'emotion_summary': pandas.DataFrame or None,
          'linguistic_correlations': pandas.DataFrame,
          'at_risk_businesses': pandas.DataFrame
        }
    """

    # -------------------------------
    # 1. Pain points from low-star reviews
    # -------------------------------
    STOPWORDS = {
        'the','and','for','that','this','with','you','have','are','but','not','was','were',
        'they','from','your','their','has','had','our','all','too','just','then','what',
        'when','there','which','will','been','into','out','about','like','theyre','dont','cant'
    }

    def _tokenize(text: str):
        if not isinstance(text, str):
            return []
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return [w for w in words if w not in STOPWORDS]

    # pick a text column
    text_col = None
    for c in ["stripped_review", "text", "raw_review"]:
        if c in df.columns:
            text_col = c
            break

    pain_points_df = pd.DataFrame(columns=["Word", "Frequency"])
    pain_point_chart = None

    if "stars" in df.columns and text_col is not None:
        low_df = df.filter(pl.col("stars") < 3.0)
        if len(low_df) > 0:
            texts = low_df.select(text_col).to_series().to_list()
            counts = {}
            for t in texts:
                for w in _tokenize(t):
                    counts[w] = counts.get(w, 0) + 1
            if counts:
                top_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
                pain_points_df = pd.DataFrame(top_items, columns=["Word", "Frequency"])
                pain_point_chart = (
                    alt.Chart(pain_points_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Frequency:Q", title="Frequency in Low-Rated Reviews"),
                        y=alt.Y("Word:N", sort="-x", title="Pain Point"),
                        tooltip=["Word", "Frequency"],
                    )
                    .properties(title="Top Pain Points", width=600, height=400)
                )

    # -------------------------------
    # 2. Emotion summary
    # -------------------------------
    emotion_cols = [
        c for c in df.columns
        if c.endswith("_pct") and any(k in c.lower() for k in ["anger", "fear", "joy", "sad", "disgust"])
    ]
    emotion_summary = None

    if emotion_cols:
        averages = {}
        for c in emotion_cols:
            try:
                averages[c] = float(df.select(pl.col(c).mean()).item() or 0.0)
            except Exception:
                averages[c] = np.nan
        if "stars" in df.columns:
            try:
                averages["avg_stars"] = float(df.select(pl.col("stars").mean()).item())
            except Exception:
                averages["avg_stars"] = np.nan
        emotion_summary = pd.DataFrame([averages])
    elif "dominant_emotion" in df.columns:
        emotion_summary = (
            df.group_by("dominant_emotion")
              .agg(pl.count())
              .sort("count", descending=True)
              .rename({"count": "review_count"})
              .to_pandas()
        )

    # -------------------------------
    # 3. Linguistic correlations with stars
    # -------------------------------
    # Dynamically detect linguistic percentage columns
    ling_cols = [
        c for c in df.columns
        if c.endswith("_pct") and any(tag in c for tag in ["noun", "verb", "adj", "adv"])
    ]

    ling_corr_df = pd.DataFrame(columns=["feature", "pearson_corr_with_stars"])

    if "stars" in df.columns and ling_cols:
        sub = df.select(["stars"] + ling_cols).to_pandas().dropna()
        if len(sub) > 0:
            corrs = sub.corr(method="pearson")["stars"].drop("stars")
            ling_corr_df = (
                corrs.reset_index()
                     .rename(columns={"index": "feature", "stars": "pearson_corr_with_stars"})
            )

    # -------------------------------
    # 4. At-risk businesses (recent decline)
    # -------------------------------
    at_risk_df = pd.DataFrame()

    if all(x in df.columns for x in ["business_id", "review_date", "stars"]):
        try:
            df_dt = df.with_columns(
                pl.col("review_date")
                  .str.strptime(pl.Date, strict=False)
                  .cast(pl.Datetime)
                  .dt.truncate("1mo")
            )
        except Exception:
            df_dt = df.with_columns(
                pl.col("review_date").cast(pl.Datetime).dt.truncate("1mo")
            )

        # latest date in this subset
        now = df_dt.select(pl.col("review_date").max()).item()
        now = pd.to_datetime(now) if now else pd.Timestamp.now()

        six_months_ago = now - pd.DateOffset(months=6)
        twelve_months_ago = now - pd.DateOffset(months=12)

        recent = df_dt.filter(pl.col("review_date") >= six_months_ago)
        prior  = df_dt.filter(
            (pl.col("review_date") >= twelve_months_ago) &
            (pl.col("review_date") < six_months_ago)
        )

        recent_stats = recent.group_by("business_id").agg([
            pl.col("stars").mean().alias("recent_mean_star"),
            (pl.col("stars") < 3.0).sum().alias("recent_low_count"),
            pl.count().alias("recent_review_count"),
        ])
        prior_stats = prior.group_by("business_id").agg([
            pl.col("stars").mean().alias("prior_mean_star"),
            (pl.col("stars") < 3.0).sum().alias("prior_low_count"),
            pl.count().alias("prior_review_count"),
        ])

        risk = recent_stats.join(prior_stats, on="business_id", how="left").to_pandas().fillna(0)
        risk["decline"] = risk["prior_mean_star"] - risk["recent_mean_star"]
        risk["recent_low_pct"] = np.where(
            risk["recent_review_count"] > 0,
            risk["recent_low_count"] / risk["recent_review_count"],
            0,
        )

        def _flag(row):
            return (row["decline"] > 0.3) or (
                row["recent_low_pct"] > 0.25 and row["recent_review_count"] >= 10
            )

        risk["at_risk_flag"] = risk.apply(_flag, axis=1)

        # attach business info
        if "business_id" in business_lookup.columns:
            risk = risk.merge(business_lookup.to_pandas(), on="business_id", how="left")

        at_risk_df = risk.sort_values(["at_risk_flag", "decline"], ascending=[False, False])

    return {
        "pain_point_chart": pain_point_chart,
        "pain_points_df": pain_points_df,
        "emotion_summary": emotion_summary,
        "linguistic_correlations": ling_corr_df,
        "at_risk_businesses": at_risk_df,
    }


# ## Prescriptive Analysis

# In[38]:


def prescriptive_analysis(df: pl.DataFrame, business_lookup: pl.DataFrame):
    """
    High-level prescriptive summary built on top of diagnostic_analysis.

    Returns
    -------
    dict
        {
          'recommendation': str,
          'action_metric': float,
          'recommended_actions': list[str],
          'business_flags': pandas.DataFrame
        }
    """
    import pandas as pd
    import numpy as np

    if df is None or len(df) == 0:
        return {
            "recommendation": "No data available for this selection.",
            "action_metric": np.nan,
            "recommended_actions": [],
            "business_flags": pd.DataFrame(),
        }

    # -----------------------------
    # 1. Overall performance level
    # -----------------------------
    if "stars" in df.columns:
        avg_stars = float(df.select(pl.col("stars").mean()).item())
    else:
        avg_stars = np.nan

    # -----------------------------
    # 2. Recent vs prior 6-month trend
    # -----------------------------
    trend_label = "stable"
    trend_delta = 0.0

    if all(c in df.columns for c in ["review_date", "stars"]):
        try:
            # parse dates to month
            try:
                df_dt = df.with_columns(
                    pl.col("review_date")
                      .str.strptime(pl.Date, strict=False)
                      .cast(pl.Datetime)
                      .dt.truncate("1mo")
                )
            except Exception:
                df_dt = df.with_columns(
                    pl.col("review_date").cast(pl.Datetime).dt.truncate("1mo")
                )

            last_date = df_dt.select(pl.col("review_date").max()).item()
            last_date = pd.to_datetime(last_date) if last_date else pd.Timestamp.now()

            six_months_ago = last_date - pd.DateOffset(months=6)
            twelve_months_ago = last_date - pd.DateOffset(months=12)

            recent = df_dt.filter(pl.col("review_date") >= six_months_ago)
            prior  = df_dt.filter(
                (pl.col("review_date") >= twelve_months_ago) &
                (pl.col("review_date") < six_months_ago)
            )

            if len(recent) > 0 and len(prior) > 0:
                recent_mean = float(recent.select(pl.col("stars").mean()).item())
                prior_mean  = float(prior.select(pl.col("stars").mean()).item())
                trend_delta = recent_mean - prior_mean

                if trend_delta > 0.1:
                    trend_label = "improving"
                elif trend_delta < -0.1:
                    trend_label = "declining"
                else:
                    trend_label = "stable"
        except Exception:
            # if anything goes wrong, leave trend as 'stable'
            pass

    # -----------------------------
    # 3. Pull at-risk businesses from diagnostic_analysis
    # -----------------------------
    diag = diagnostic_analysis(df, business_lookup)
    at_risk = diag.get("at_risk_businesses", pd.DataFrame())

    # -----------------------------
    # 4. Build recommendation text & actions
    # -----------------------------
    if np.isnan(avg_stars):
        level_msg = "Overall performance level is unclear because star ratings are missing."
    elif avg_stars >= 4.2:
        level_msg = f"Overall performance is strong with an average rating of about {avg_stars:.2f} stars."
    elif avg_stars >= 3.6:
        level_msg = f"Overall performance is moderate with an average rating of about {avg_stars:.2f} stars."
    else:
        level_msg = f"Overall performance is weak with an average rating of about {avg_stars:.2f} stars."

    if trend_label == "improving":
        trend_msg = f"Ratings have been improving recently (change of about {trend_delta:+.2f} stars over the last 6 months)."
    elif trend_label == "declining":
        trend_msg = f"Ratings have been declining recently (change of about {trend_delta:+.2f} stars over the last 6 months)."
    else:
        trend_msg = "Ratings are relatively stable over the last 6 months."

    if at_risk is not None and not at_risk.empty:
        risk_msg = (
            f"There are {at_risk['at_risk_flag'].sum()} businesses flagged as 'at risk' "
            f"based on recent rating declines or high low-star review rates."
        )
    else:
        risk_msg = "No businesses are strongly flagged as 'at risk' based on recent review trends."

    recommendation = " ".join([level_msg, trend_msg, risk_msg])

    # Suggested concrete actions
    actions = []
    if avg_stars < 4.0:
        actions.append("Prioritize service recovery for recent 1–2 star reviews.")
        actions.append("Review operational processes at locations with recurring low-star feedback.")
    if trend_label == "declining":
        actions.append("Investigate changes in staffing, pricing, or policies over the last 6–12 months.")
        actions.append("Launch a targeted campaign to solicit feedback from loyal customers.")
    if at_risk is not None and not at_risk.empty:
        actions.append("Create mitigation plans for flagged 'at risk' businesses and monitor them weekly.")
    if not actions:
        actions.append("Maintain current best practices and continue monitoring sentiment monthly.")

    return {
        "recommendation": recommendation,
        "action_metric": avg_stars,
        "recommended_actions": actions,
        "business_flags": at_risk,
    }


# ## Master Function - Master Function Scaffold

# ### Streamlit Page Setup & User Input

# In[39]:


#5. Master Function Scaffold
def yelp_dashboard():
    """
    Master function for the Yelp Universal Business Dashboard.

    High-level flow:
      1. Configure Streamlit page and sidebar inputs.
      2. Load & preprocess data for the chosen industry.
      3. Apply user filters to create the analysis subset.
      4. Run analytics (Descriptive, Predictive, Diagnostic, Prescriptive).
      5. Build dashboard layout & display (to be added later).
    """

    # ---------------------------------------------------------
    # 1. Streamlit Page Setup & User Input
    # ---------------------------------------------------------
    st.set_page_config(
        page_title="Yelp Universal Business Dashboard",
        layout="wide",
    )

    st.sidebar.title("Yelp Universal Dashboard")

    # Industry selection (required)
    industry_choice = st.sidebar.selectbox(
        "Select Industry",
        options=["Hair", "Mexican Food"],
        index=0,
    )

    # Optional business-level filters
    st.sidebar.markdown("### Filters (optional)")
    business_id = st.sidebar.text_input("Business ID (exact match)")
    business_name = st.sidebar.text_input("Business Name (contains)")
    city = st.sidebar.text_input("City")
    state = st.sidebar.text_input("State (e.g., CA, NY)")

    # Forecast horizon for predictive analysis
    periods = st.sidebar.slider(
        "Months to Forecast",
        min_value=3,
        max_value=12,
        value=6,
        step=1,
    )

    # Diagnostic – number of pain-point keywords
    top_n_pain_points = st.sidebar.slider(
        "Number of Pain-Point Keywords",
        min_value=5,
        max_value=30,
        value=20,
        step=5,
    )

    # ---------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # ---------------------------------------------------------
    st.header("Data Loading & Preprocessing")
    st.subheader("Loading Data...")

    try:
        df, business_lookup = load_and_preprocess_data(industry_choice)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    st.success(f"Data loaded successfully! Total reviews: {len(df)}")

    # ---------------------------------------------------------
    # 3. Apply User Filters
    # ---------------------------------------------------------
    st.header("Filter Selection & Subset Creation")

    df_filtered, subset_label = apply_user_filters(
        df_full=df,
        business_lookup=business_lookup,
        business_id_filter=business_id or None,
        business_name=business_name or None,
        state_filter=state or None,
        city_filter=city or None,
    )

    st.write(f"Current slice: **{subset_label}**")
    st.write(f"Number of reviews in slice: **{len(df_filtered)}**")

    if len(df_filtered) == 0:
        st.warning("No data available for the selected filters. Adjust your filters and try again.")
        return

    # Simple overall metrics (we'll use these in the layout later)
    try:
        avg_star_overall = float(df_filtered.select(pl.col("stars").mean()).item())
    except Exception:
        avg_star_overall = float("nan")

    try:
        n_businesses = int(df_filtered.select(pl.col("business_id").n_unique()).item())
    except Exception:
        n_businesses = 0

    # ---------------------------------------------------------
    # 4. Run Analytics (Descriptive, Predictive, Diagnostic, Prescriptive)
    # ---------------------------------------------------------
    st.header("Running Analytics...")

    # 4.1 Descriptive analysis (Alex)
    line_chart, pie_chart, top_5_best, top_5_worst = descriptive_analysis(
        df_filtered,
        business_lookup,
    )

    # 4.2 Predictive analysis (Alex)
    forecast_chart, predicted_star = predictive_analysis(
        df_filtered,
        periods=periods,
    )

    # 4.3 Diagnostic analysis (Eddie)
    diag_results = diagnostic_analysis(
        df_filtered,
        business_lookup,
        top_n=top_n_pain_points,
    )

    # 4.4 Prescriptive analysis (Eddie)
    presc_results = prescriptive_analysis(
        df_filtered,
        business_lookup,
    )

    st.success("Analytics completed successfully.")

        # ---------------------------------------------------------
    # 5. Dashboard Layout & Display
    # ---------------------------------------------------------

    # ---- Top of dashboard ----
    st.title("Yelp Business Health Dashboard")
    st.subheader(f"Industry: {industry_choice}")
    st.caption(
        f"Current slice: **{subset_label}**  |  "
        f"Reviews in slice: {len(df_filtered)}  |  "
        f"Distinct businesses: {n_businesses}"
    )

    # Top-level metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        if avg_star_overall == avg_star_overall:  # not NaN
            st.metric("Average Star Rating (Current Slice)",
                      f"{avg_star_overall:.2f}")
        else:
            st.metric("Average Star Rating (Current Slice)", "N/A")
    with col2:
        st.metric(f"Forecasted Stars in {periods} Months",
                  f"{predicted_star:.2f}")
    with col3:
        st.metric("Number of Businesses", n_businesses)

    # ---------------------------------------------------------
    # 5.1 Descriptive Analysis
    # ---------------------------------------------------------
    st.header("1. Descriptive Analysis")

    st.subheader("Star Ratings Over Time")
    if line_chart is not None:
        st.altair_chart(line_chart, use_container_width=True)
    else:
        st.info("Time series not available for this subset.")

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Top 5 Best Locations")
        if top_5_best is not None:
            try:
                st.dataframe(top_5_best.to_pandas())
            except AttributeError:
                st.dataframe(top_5_best)
        else:
            st.info("No best-location summary available.")

    with cols[1]:
        st.subheader("Top 5 Worst Locations")
        if top_5_worst is not None:
            try:
                st.dataframe(top_5_worst.to_pandas())
            except AttributeError:
                st.dataframe(top_5_worst)
        else:
            st.info("No worst-location summary available.")

    st.subheader("Emotion Distribution")
    if pie_chart is not None:
        st.altair_chart(pie_chart, use_container_width=True)
    else:
        st.info("Emotion distribution not available for this subset.")

    # ---------------------------------------------------------
    # 5.2 Predictive Analysis
    # ---------------------------------------------------------
    st.header("2. Predictive Analysis")

    if forecast_chart is not None:
        st.subheader("Star Rating Forecast")
        st.altair_chart(forecast_chart, use_container_width=True)
    else:
        st.info("Not enough historical data to fit a forecast model.")

    st.markdown(
        f"**Predicted average star rating in {periods} months:** "
        f"{predicted_star:.2f}"
    )

    # ---------------------------------------------------------
    # 5.3 Diagnostic Insights
    # ---------------------------------------------------------
    st.header("3. Diagnostic Insights")

    # Pain points in low-star reviews
    st.subheader("Pain Points in Low-Star Reviews")
    pain_chart = diag_results.get("pain_point_chart")
    pain_df = diag_results.get("pain_points_df")
    if pain_chart is not None:
        st.altair_chart(pain_chart, use_container_width=True)
    elif pain_df is not None and not getattr(pain_df, "empty", False):
        st.dataframe(pain_df.head(30))
    else:
        st.info("Not enough low-star review text to extract pain points.")

    # Emotion summary
    st.subheader("Emotion Summary")
    emo = diag_results.get("emotion_summary")
    if emo is not None and not getattr(emo, "empty", False):
        st.dataframe(emo)
    else:
        st.info("Emotion summary not available for this subset.")

    # Linguistic correlations
    st.subheader("Linguistic Correlations with Star Ratings")
    ling = diag_results.get("linguistic_correlations")
    if ling is not None and not getattr(ling, "empty", False):
        st.dataframe(ling)
    else:
        st.info("Not enough variation in linguistic features to compute correlations.")

    # At-risk businesses
    st.subheader("At-Risk Businesses (Recent Decline)")
    risk_df = diag_results.get("at_risk_businesses")
    if risk_df is not None and not getattr(risk_df, "empty", False):
        # Show only top 30 rows for readability
        st.dataframe(risk_df.head(30))
    else:
        st.info("No clear 'at risk' businesses based on recent trends in this subset.")

    # ---------------------------------------------------------
    # 5.4 Prescriptive Recommendations
    # ---------------------------------------------------------
    st.header("4. Prescriptive Recommendations")

    st.subheader("High-Level Recommendation")
    st.markdown(presc_results.get("recommendation", "No recommendation text available."))

    action_metric = presc_results.get("action_metric")
    if action_metric is not None:
        st.metric("Overall Action Metric (Avg Stars)", f"{action_metric:.2f}")

    st.subheader("Recommended Actions")
    rec_actions = presc_results.get("recommended_actions", [])
    if rec_actions:
        for action in rec_actions:
            st.markdown(f"- {action}")
    else:
        st.info("No specific recommended actions generated.")

    st.subheader("Flagged Businesses (for Follow-up)")
    flagged = presc_results.get("business_flags")
    if flagged is not None and not getattr(flagged, "empty", False):
        st.dataframe(flagged.head(30))
    else:
        st.info("No businesses are currently flagged for prescriptive follow-up.")


# ## Entry Point / Function Call

# In[40]:


# ---------------------------------------------------------
# 6. Entry Point / Function Call
# ---------------------------------------------------------

# NOTE:
# In a .py Streamlit app you would run:
#     streamlit run yelp_universal_dashboard.py
# which triggers this entry point.
# In Jupyter Notebook, this will still run the dashboard inline,
# though Streamlit will display "ScriptRunContext" warnings (normal).

if __name__ == "__main__":
    # Entry point: launch the Yelp Universal Business Dashboard
    yelp_dashboard()


# ## CODE TESTS

# In[41]:


# TEST 1: Load updated data for both industries

#Hair
df_hair, lookup_hair = load_and_preprocess_data("Hair")
print("Hair data shape:", df_hair.shape if df_hair is not None else "No data loaded")
print("Hair lookup preview:")
print(lookup_hair.head() if lookup_hair is not None else "No lookup loaded")

print("\n" + "-"*60 + "\n")

#Mexican Food
df_mex, lookup_mex = load_and_preprocess_data("Mexican Food")
print("Mexican data shape:", df_mex.shape if df_mex is not None else "No data loaded")
print("Mexican lookup preview:")
print(lookup_mex.head() if lookup_mex is not None else "No lookup loaded")


# In[42]:


# TEST 2: Apply a simple business-name filter on Hair data

# Make sure df_hair and lookup_hair exist from TEST 1
if df_hair is None or lookup_hair is None:
    print("Run Test 1 first so df_hair and lookup_hair are defined.")
else:
    sample_name = lookup_hair.select('name').to_series().to_list()[0]
    print("Sample business name:", sample_name)

    df_subset, label = apply_user_filters(
        df_full=df_hair,
        business_lookup=lookup_hair,
        business_id_filter=None,
        business_name=sample_name,
        state_filter=None,
        city_filter=None
    )

    print("Subset label:", label)
    print("Subset shape:", df_subset.shape)


# ## Descriptive Analysis Tests

# In[43]:


#TEST 3: Descriptive Analysis
df_hair, lookup_hair = load_and_preprocess_data("Hair")
line_chart, pie_chart, best, worst = descriptive_analysis(df_hair, lookup_hair)
line_chart


# In[44]:


# Inspect all "star" / "weight" columns in the updated data
[c for c in df_hair.columns if "star" in c.lower() or "weight" in c.lower() or "tdw" in c.lower()]


# In[45]:


#TEST #4: Top 5 Best & Top 5 Worst
line_chart, pie_chart, best, worst = descriptive_analysis(df_hair, lookup_hair)

print("Top 5 Best Locations:")
print(best)

print("\nTop 5 Worst Locations:")
print(worst)


# ## Predictive Analysis Tests

# In[46]:


[c for c in df_hair.columns if "_pct" in c.lower()]


# In[47]:


[c for c in df_hair.columns if "emotion" in c.lower()]


# In[48]:


#TEST #5: Predictive Analysis Forecast
forecast_chart, predicted_star = predictive_analysis(df_hair, periods=6)
print("Predicted star after 6 months:", predicted_star)
forecast_chart


# ## Diagnostic Analytic Test

# In[49]:


# Make sure data is loaded
df_hair, lookup_hair = load_and_preprocess_data("Hair")

# Run Eddie's diagnostic analysis on the Hair subset
results = diagnostic_analysis(df_hair, lookup_hair)
results.keys()


# ## Prescriptive Analytic Test

# In[50]:


df_hair, lookup_hair = load_and_preprocess_data("Hair")

presc_results = prescriptive_analysis(df_hair, lookup_hair)

print("High-level recommendation:")
print(presc_results["recommendation"])
print("\nAction metric:", presc_results["action_metric"])
print("\nRecommended actions:")
for a in presc_results["recommended_actions"]:
    print("-", a)

print("\nFirst few flagged businesses:")
print(presc_results["business_flags"].head())


# In[ ]:




