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

# In[124]:


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

# In[125]:


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

# In[126]:


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

# In[127]:


def predictive_analysis(df: pl.DataFrame, periods: int = 6):
    """
    Predictive analysis: time-series forecast of average star rating.

    Parameters
    ----------
    df : pl.DataFrame
        Filtered review-level data for the current slice.
    periods : int
        Number of future months to forecast.

    Returns
    -------
    forecast_chart : alt.Chart or None
        Line chart showing historical and forecasted star ratings.
    info : dict
        {
            'predicted_star': float,
            'forecast_horizon_months': int
        }
    """
    
    # Guard rails: need date and stars
    if "review_date" not in df.columns or "stars" not in df.columns:
        return None, {"predicted_star": np.nan, "forecast_horizon_months": periods}

    # Convert to pandas and ensure proper datetime / sort
    pdf = df.select(["review_date", "stars"]).to_pandas()
    pdf["review_date"] = pd.to_datetime(pdf["review_date"])
    pdf = pdf.sort_values("review_date")

    # Monthly average star rating
    monthly = (
        pdf.set_index("review_date")["stars"]
        .resample("MS")  # Month start frequency
        .mean()
        .ffill()
    )

    # If we do not have enough history, fall back to trend only
    if len(monthly) < 6:
        hist_df = monthly.reset_index()
        hist_df.columns = ["date", "avg_stars"]

        chart = (
            alt.Chart(hist_df)
            .mark_line()
            .encode(
                x="date:T",
                y=alt.Y("avg_stars:Q", title="Average Star Rating"),
                tooltip=["date:T", "avg_stars:Q"],
            )
            .properties(
                title="Star Rating Trend (insufficient history for robust forecast)"
            )
        )

        last_val = float(hist_df["avg_stars"].iloc[-1])
        return chart, {
            "predicted_star": last_val,
            "forecast_horizon_months": periods,
        }

    # Fit a simple ARIMA model
    model = ARIMA(monthly, order=(1, 1, 1))
    fitted = model.fit()

    # Forecast future months
    forecast_values = fitted.forecast(steps=periods)

    last_date = monthly.index[-1]
    future_index = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS",
    )

    forecast_series = pd.Series(forecast_values.values, index=future_index)

    # Build data for plotting
    hist_df = monthly.reset_index()
    hist_df.columns = ["date", "avg_stars"]

    forecast_df = forecast_series.reset_index()
    forecast_df.columns = ["date", "forecast_star"]

    hist_plot = hist_df[["date", "avg_stars"]].copy()
    hist_plot["series"] = "Historical"
    hist_plot = hist_plot.rename(columns={"avg_stars": "star"})

    forecast_plot = forecast_df[["date", "forecast_star"]].copy()
    forecast_plot["series"] = "Forecast"
    forecast_plot = forecast_plot.rename(columns={"forecast_star": "star"})

    combined = pd.concat([hist_plot, forecast_plot], ignore_index=True)

    forecast_chart = (
        alt.Chart(combined)
        .mark_line()
        .encode(
            x="date:T",
            y=alt.Y("star:Q", title="Average Star Rating"),
            color=alt.Color("series:N", title="Series"),
            tooltip=["date:T", "star:Q", "series:N"],
        )
        .properties(title=f"Star Rating Forecast ({periods} Months Ahead)")
    )

    predicted_star = float(forecast_plot["star"].iloc[-1])

    return forecast_chart, {
        "predicted_star": predicted_star,
        "forecast_horizon_months": periods,
    }


# ## Diagnostic Analysis

# In[128]:


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
    import pandas as pd
    import numpy as np
    import altair as alt
    import re

    # -------------------------------------------------------
    # 1) Pain points from low-star reviews
    # -------------------------------------------------------
    STOPWORDS = {
        "the", "and", "for", "that", "this", "with", "you", "have", "are", "but", "not",
        "was", "were", "they", "from", "your", "their", "has", "had", "our", "all", "too",
        "just", "then", "what", "when", "there", "which", "will", "been", "into", "out",
        "about", "like", "theyre", "dont", "cant",
        # extra generic verbs we do not want as pain points
        "went", "got"
    }

    def _tokenize(text: str):
        if not isinstance(text, str):
            return []
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return [w for w in words if w not in STOPWORDS and w not in {"went", "got"}]

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
                        tooltip=["Word", "Frequency"]
                    )
                    .properties(title="Top Pain Points", width=600, height=400)
                )

    # -------------------------------------------------------
    # 2) Emotion summary
    # -------------------------------------------------------
    emotion_cols = [
        c for c in df.columns
        if c.endswith("_int_avg") or (
            c.endswith("_pct") and any(k in c for k in ["anger","fear","joy","sad","disgust"])
        )
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

    # -------------------------------------------------------
    # 3) Linguistic correlations with stars
    #     (this is the part we are FIXING)
    # -------------------------------------------------------
    # Use the averaged linguistic features created earlier
    ling_cols = [
        c for c in df.columns
        if c in ["avg_noun_pct", "avg_verb_pct", "avg_adj_pct", "avg_adv_pct"]
    ]

    ling_corr_df = pd.DataFrame(columns=["feature", "pearson_corr_with_stars"])

    if "stars" in df.columns and ling_cols:
        sub = df.select(["stars"] + ling_cols).to_pandas().dropna()

        if len(sub) > 0:
            # Drop any columns with no variation (constant values)
            usable_ling_cols = [
                c for c in ling_cols
                if sub[c].nunique(dropna=True) > 1
            ]

            if usable_ling_cols:
                corrs = sub[["stars"] + usable_ling_cols].corr(method="pearson")["stars"]
                corrs = corrs.drop(labels=["stars"])
                ling_corr_df = (
                    corrs.reset_index()
                         .rename(columns={"index": "feature",
                                          "stars": "pearson_corr_with_stars"})
                )
            # If no usable columns, ling_corr_df stays as an empty DataFrame

    # -------------------------------------------------------
    # 4) At-risk businesses (same logic as before)
    # -------------------------------------------------------
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

        now = df_dt.select(pl.col("review_date").max()).item()
        now = pd.to_datetime(now) if now else pd.Timestamp.now()

        six_months_ago = now - pd.DateOffset(months=6)
        twelve_months_ago = now - pd.DateOffset(months=12)

        recent = df_dt.filter(pl.col("review_date") >= six_months_ago)
        prior  = df_dt.filter(
            (pl.col("review_date") >= twelve_months_ago) &
            (pl.col("review_date") < six_months_ago)
        )

        recent_stats = recent.group_by("business_id").agg(
            [
                pl.col("stars").mean().alias("recent_mean_star"),
                (pl.col("stars") < 3.0).sum().alias("recent_low_count"),
                pl.count().alias("recent_review_count")
            ]
        )
        
        prior_stats = prior.group_by("business_id").agg(
            [
                pl.col("stars").mean().alias("prior_mean_star"),
                (pl.col("stars") < 3.0).sum().alias("prior_low_count"),
                pl.count().alias("prior_review_count")
            ]
        )

        risk = recent_stats.join(prior_stats, on="business_id", how="left").to_pandas().fillna(0)
        risk["decline"] = risk["prior_mean_star"] - risk["recent_mean_star"]
        risk["recent_low_pct"] = np.where(
            risk["recent_review_count"] > 0,
            risk["recent_low_count"] / risk["recent_review_count"],
            0
        )

        def _flag(row):
            return (row["decline"] > 0.3) or (
                row["recent_low_pct"] > 0.25 and row["recent_review_count"] >= 10
            )

        risk["at_risk_flag"] = risk.apply(_flag, axis=1)

        if "business_id" in business_lookup.columns:
            risk = risk.merge(business_lookup.to_pandas(), on="business_id", how="left")

        at_risk_df = risk.sort_values(["at_risk_flag", "decline"], ascending=[False, False])

    # -------------------------------------------------------
    # Return dictionary
    # -------------------------------------------------------
    return {
        "pain_point_chart": pain_point_chart,
        "pain_points_df": pain_points_df,
        "emotion_summary": emotion_summary,
        "linguistic_correlations": ling_corr_df,
        "at_risk_businesses": at_risk_df
    }


# ## Prescriptive Analysis

# In[129]:


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

# In[130]:


#5. Master Function Scaffold
def yelp_dashboard():
    """
    Master function for the Yelp Universal Business Dashboard.
    Relies on the following helper functions defined elsewhere:
      - load_and_preprocess_data(industry)
      - apply_user_filters(df_full, business_lookup, business_id_filter=None,
                           business_name=None, city_filter=None, state_filter=None)
      - descriptive_analysis(df, business_lookup)
      - predictive_analysis(df, periods)
      - diagnostic_analysis(df, business_lookup, top_n=20)
      - prescriptive_analysis(df, business_lookup)
    """

    import pandas as pd
    import altair as alt

    # ------------------------------------------------------------------
    # 1. Streamlit Page Setup & User Input
    # ------------------------------------------------------------------
    st.set_page_config(
        page_title="Yelp Universal Business Dashboard",
        layout="wide",
    )

    st.sidebar.title("Yelp Universal Dashboard")

    industry_choice = st.sidebar.selectbox(
        "Select Industry",
        options=["Hair", "Mexican Food"],
        index=0,
    )

    st.sidebar.markdown("### Filters (optional)")
    business_id = st.sidebar.text_input("Business ID (exact match)")
    business_name = st.sidebar.text_input("Business Name (contains)")
    city = st.sidebar.text_input("City")
    state = st.sidebar.text_input("State (e.g., CA, NY)")

    periods = st.sidebar.slider(
        "Months to Forecast",
        min_value=3,
        max_value=12,
        value=6,
        step=1,
    )

    top_n_pain_points = st.sidebar.slider(
        "Number of Pain-Point Keywords",
        min_value=5,
        max_value=30,
        value=20,
        step=5,
    )

    st.title("Yelp Business Health Dashboard")

    # ------------------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # ------------------------------------------------------------------
    st.subheader("Data Loading & Preprocessing")
    st.subheader("Loading Data...")

    try:
        df, business_lookup = load_and_preprocess_data(industry_choice)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    st.success(f"Data loaded successfully. Total reviews: {len(df)}")

    # ------------------------------------------------------------------
    # 3. Apply User Filters
    # ------------------------------------------------------------------
    st.subheader("Filter Selection & Subset Creation")

    try:
        df_filtered, subset_label = apply_user_filters(
            df_full=df,
            business_lookup=business_lookup,
            business_id_filter=business_id or None,
            business_name=business_name or None,
            city_filter=city or None,
            state_filter=state or None,
        )
    except TypeError:
        # Fallback if apply_user_filters has a positional-only signature
        df_filtered, subset_label = apply_user_filters(
            df,
            business_lookup,
            business_id or None,
            business_name or None,
            city or None,
            state or None,
        )

    if df_filtered is None or len(df_filtered) == 0:
        st.warning("No reviews match the selected filters.")
        return

    st.write(f"Current slice: {subset_label}")
    st.write(f"Number of reviews in slice: {len(df_filtered)}")

    # ------------------------------------------------------------------
    # 4. Run Analytics
    # ------------------------------------------------------------------
    st.subheader("Running Analytics...")

    try:
        desc_results = descriptive_analysis(df_filtered, business_lookup)
    except Exception:
        desc_results = None

    try:
        pred_results = predictive_analysis(df_filtered, periods)
    except Exception:
        pred_results = None

    try:
        diag_results = diagnostic_analysis(
            df_filtered, business_lookup, top_n=top_n_pain_points
        )
    except Exception:
        diag_results = None

    try:
        presc_results = prescriptive_analysis(df_filtered, business_lookup)
    except Exception:
        presc_results = None

    st.success("Analytics completed successfully.")

    # Helper: safe conversion to pandas
    def _to_pandas_safe(obj):
        if obj is None:
            return None
        if isinstance(obj, pd.DataFrame):
            return obj
        if hasattr(obj, "to_pandas"):
            try:
                return obj.to_pandas()
            except Exception:
                pass
        return obj

    # ------------------------------------------------------------------
    # 5. Dashboard Layout & Display
    # ------------------------------------------------------------------

    st.header(f"Industry: {industry_choice}")
    st.write(
        f"Current slice: {subset_label} | "
        f"Reviews in slice: {len(df_filtered)}"
    )

    # --------------------------------------------------------------
    # 5.1 Descriptive Analytics
    # --------------------------------------------------------------
    st.header("Descriptive Analytics")

    rating_trend_chart = None
    pie_chart = None
    best_df = None
    worst_df = None

    # Support both dict and tuple return formats from descriptive_analysis
    if isinstance(desc_results, dict):
        rating_trend_chart = desc_results.get("rating_trend_chart") or desc_results.get(
            "line_chart"
        )
        pie_chart = desc_results.get("emotion_pie") or desc_results.get("pie_chart")
        best_df = desc_results.get("best_locations")
        worst_df = desc_results.get("worst_locations")
    elif isinstance(desc_results, (list, tuple)):
        if len(desc_results) >= 1:
            rating_trend_chart = desc_results[0]
        if len(desc_results) >= 2:
            pie_chart = desc_results[1]
        if len(desc_results) >= 3:
            best_df = desc_results[2]
        if len(desc_results) >= 4:
            worst_df = desc_results[3]

    # Line chart for actual vs time-decay-weighted stars
    if rating_trend_chart is not None:
        st.altair_chart(rating_trend_chart, use_container_width=True)
    else:
        st.info("No rating trend chart available for this slice.")

    # Emotion pie chart with percent and count labels
    st.subheader("Emotion Distribution")

    emotion_chart_rendered = False

    if "dominant_emotion" in df_filtered.columns:
        emo_df = (
            df_filtered.to_pandas()["dominant_emotion"]
            .dropna()
            .value_counts()
            .rename_axis("emotion")
            .reset_index(name="count")
        )

        if len(emo_df) > 0:
            total = emo_df["count"].sum()
            emo_df["percent"] = emo_df["count"] / total * 100.0
            emo_df["label"] = emo_df.apply(
                lambda r: f"{r['emotion']} ({r['count']}, {r['percent']:.1f}%)",
                axis=1,
            )

            base = alt.Chart(emo_df)

            pie = base.mark_arc().encode(
                theta=alt.Theta("count:Q", title="Number of Reviews"),
                color=alt.Color("emotion:N", title="Emotion"),
            )

            text = base.mark_text(radius=100, size=11).encode(
                text=alt.Text("label:N")
            )

            pie_chart_with_labels = pie + text

            st.altair_chart(pie_chart_with_labels, use_container_width=True)
            emotion_chart_rendered = True

    # Fallback to any pie chart returned from descriptive_analysis
    if not emotion_chart_rendered:
        if pie_chart is not None:
            st.altair_chart(pie_chart, use_container_width=True)
        else:
            st.info("No emotion data available for this slice.")

    # Top 5 best/worst locations – updated to show Business Name + City, State
    best_df = _to_pandas_safe(best_df)
    worst_df = _to_pandas_safe(worst_df)

    best_display = None
    worst_display = None

    if best_df is not None and len(best_df) > 0:
        best_display = best_df.copy()
        if "state" in best_display.columns:
            best_display["Location"] = best_display.apply(
                lambda r: f"{r['name']} ({r['city']}, {r['state']})",
                axis=1,
            )
        else:
            best_display["Location"] = best_display.apply(
                lambda r: f"{r['name']} ({r['city']})",
                axis=1,
            )
        cols = ["Location", "avg_stars", "review_count"]
        best_display = best_display[[c for c in cols if c in best_display.columns]]

    if worst_df is not None and len(worst_df) > 0:
        worst_display = worst_df.copy()
        if "state" in worst_display.columns:
            worst_display["Location"] = worst_display.apply(
                lambda r: f"{r['name']} ({r['city']}, {r['state']})",
                axis=1,
            )
        else:
            worst_display["Location"] = worst_display.apply(
                lambda r: f"{r['name']} ({r['city']})",
                axis=1,
            )
        cols = ["Location", "avg_stars", "review_count"]
        worst_display = worst_display[[c for c in cols if c in worst_display.columns]]

    col1, col2 = st.columns(2)

    with col1:
        if best_display is not None and len(best_display) > 0:
            st.markdown("Top 5 Best Locations")
            st.dataframe(best_display)

    with col2:
        if worst_display is not None and len(worst_display) > 0:
            st.markdown("Top 5 Worst Locations")
            st.dataframe(worst_display)

    st.markdown("---")

    # --------------------------------------------------------------
    # 5.2 Predictive Analytics
    # --------------------------------------------------------------
    st.header("Predictive Analytics")

    forecast_chart = None
    forecast_value = None

    if isinstance(pred_results, dict):
        forecast_chart = pred_results.get("forecast_chart") or pred_results.get("chart")
        forecast_value = (
            pred_results.get("forecast_value")
            or pred_results.get("predicted_avg")
            or pred_results.get("pred_avg")
        )
    elif isinstance(pred_results, (list, tuple)):
        if len(pred_results) >= 1:
            forecast_chart = pred_results[0]
        if len(pred_results) >= 2 and isinstance(pred_results[1], dict):
            meta = pred_results[1]
            forecast_value = (
                meta.get("forecast_value")
                or meta.get("predicted_avg")
                or meta.get("pred_avg")
            )
    else:
        forecast_chart = pred_results

    if forecast_chart is not None:
        st.altair_chart(forecast_chart, use_container_width=True)
    else:
        st.info("No forecast chart available for this slice.")

    if forecast_value is not None and not isinstance(
        forecast_value, (dict, list, tuple)
    ):
        try:
            fv = float(forecast_value)
            st.write(
                f"Predicted average star rating after {periods} months: {fv:.3f}"
            )
        except (TypeError, ValueError):
            pass

    st.markdown("---")

    # --------------------------------------------------------------
    # 5.3 Diagnostic Analytics
    # --------------------------------------------------------------
    st.header("Diagnostic Analytics")

    if isinstance(diag_results, dict):
        pain_chart = diag_results.get("pain_point_chart")
        pain_df = _to_pandas_safe(diag_results.get("pain_points_df"))
        emotion_summary = _to_pandas_safe(diag_results.get("emotion_summary"))
        ling_corr = _to_pandas_safe(diag_results.get("linguistic_correlations"))
        at_risk_df = _to_pandas_safe(diag_results.get("at_risk_businesses"))
    else:
        pain_chart = None
        pain_df = None
        emotion_summary = None
        ling_corr = None
        at_risk_df = None

    st.subheader("Top Pain-Point Keywords (Low-Star Reviews)")
    if pain_chart is not None:
        st.altair_chart(pain_chart, use_container_width=True)
    elif pain_df is not None and len(pain_df) > 0:
        st.dataframe(pain_df)
    else:
        st.info("No pain-point keyword data available for this slice.")

    st.subheader("Emotion Summary (if available)")
    if emotion_summary is not None and len(emotion_summary) > 0:
        st.dataframe(emotion_summary)
    else:
        st.info("No emotion summary available for this slice.")

    st.subheader("Linguistic Features vs Star Ratings")
    if ling_corr is not None and len(ling_corr) > 0:
        st.dataframe(ling_corr)
    else:
        st.info(
            "Linguistic correlations are not available for this slice "
            "(not enough variation in linguistic features)."
        )

    st.subheader("Businesses Flagged as At Risk")
    if at_risk_df is not None and len(at_risk_df) > 0:
        st.dataframe(at_risk_df)
    else:
        st.info("No businesses are currently flagged as at risk for this slice.")

    st.markdown("---")

    # --------------------------------------------------------------
    # 5.4 Prescriptive Recommendations
    # --------------------------------------------------------------
    st.header("Prescriptive Recommendations")

    if isinstance(presc_results, dict):
        recommendation_text = presc_results.get("recommendation_text") or presc_results.get(
            "recommendation"
        )
        action_metric = presc_results.get("action_metric")
        flagged = _to_pandas_safe(presc_results.get("business_flags"))
    else:
        recommendation_text = None
        action_metric = None
        flagged = None

    st.subheader("High-Level Recommendation")
    st.markdown(
        recommendation_text
        or "No recommendation text available for this slice."
    )

    st.subheader("Overall Action Metric (Avg Stars)")
    if action_metric is not None:
        try:
            st.metric(
                "Overall Action Metric (Avg Stars)",
                f"{float(action_metric):.2f}",
            )
        except (TypeError, ValueError):
            st.write(action_metric)
    else:
        st.info("No action metric available for this slice.")

    st.subheader("Flagged Businesses (for Follow-up)")
    if flagged is not None and len(flagged) > 0:
        st.dataframe(flagged.head(30))
    else:
        st.info("No businesses are currently flagged for prescriptive follow-up.")


# ## Entry Point / Function Call

# In[131]:


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

# In[132]:


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


# In[133]:


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

# In[134]:


#TEST 3: Descriptive Analysis
df_hair, lookup_hair = load_and_preprocess_data("Hair")
line_chart, pie_chart, best, worst = descriptive_analysis(df_hair, lookup_hair)
line_chart


# In[135]:


# Inspect all "star" / "weight" columns in the updated data
[c for c in df_hair.columns if "star" in c.lower() or "weight" in c.lower() or "tdw" in c.lower()]


# In[136]:


#TEST #4: Top 5 Best & Top 5 Worst
line_chart, pie_chart, best, worst = descriptive_analysis(df_hair, lookup_hair)

print("Top 5 Best Locations:")
print(best)

print("\nTop 5 Worst Locations:")
print(worst)


# ## Predictive Analysis Tests

# In[137]:


[c for c in df_hair.columns if "_pct" in c.lower()]


# In[138]:


[c for c in df_hair.columns if "emotion" in c.lower()]


# #TEST #5: Predictive Analysis Forecast
# forecast_chart, predicted_star = predictive_analysis(df_hair, periods=6)
# print("Predicted star after 6 months:", predicted_star)
# forecast_chart

# ## Diagnostic Analytic Test

# In[140]:


# Make sure data is loaded
df_hair, lookup_hair = load_and_preprocess_data("Hair")

# Run Eddie's diagnostic analysis on the Hair subset
results = diagnostic_analysis(df_hair, lookup_hair)
results.keys()


# ## Prescriptive Analytic Test

# In[141]:


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




