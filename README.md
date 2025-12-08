# Yelp Universal Business Dashboard  
**Streamlit-Based Analytics Application for MSBA 503 – Programming Analytics II**  
Team: **Alex Snyder, Eddie, Lily**
Semester: Fall 2025  

---

##  Overview  
The **Yelp Universal Business Dashboard** is an interactive analytics application built using **Python**, **Polars**, **Pandas**, and **Streamlit**, designed to analyze Yelp business review data across multiple industries (e.g., **Hair**, **Mexican Food**). The dashboard automates:

- **Descriptive analytics** (KPIs, trends, best/worst locations)  
- **Diagnostic analytics** (pain-point keyword extraction, linguistic indicators, emotion analysis)  
- **Predictive analytics** (time-series forecasting using ARIMA)  
- **Prescriptive analytics** (actionable recommendations + at-risk business detection)

This project integrates advanced data processing, natural language analysis, and business intelligence principles into an easy-to-use web application.

---

## Key Features  

### **1. Descriptive Analytics **
- Monthly star-rating trends (actual vs. time-decayed weighted rating)  
- Top 5 best-performing business locations  
- Top 5 lowest-performing business locations  
- High-level KPIs:  
  - Total reviews  
  - Distinct businesses  
  - Average rating  

---

### **2. Diagnostic Analytics **
- Identification of **pain-point keywords** using frequency analysis  
- Emotion scoring with NRC lexicon (anger, joy, fear, trust, etc.)  
- Linguistic correlation metrics (negativity %, subjectivity %)  
- Business-level diagnostic flags (recurring low-star patterns, declining trends, etc.)

---

### **3. Predictive Analytics **
- **ARIMA time-series forecasting**  
- Forecast horizon adjustable from the sidebar (3–12 months)  
- Predicts expected future average star ratings  
- Supports business-level or industry-level forecasting  

---

### **4. Prescriptive Analytics **
Uses trends + diagnostics to determine:

- High-level recommendation summary  
- Action metric (avg weighted star score)  
- Recommended operational actions  
- Flagged businesses “**at risk**” based on:  
  - Recent decline in average stars  
  - High low-star review rate  
  - Elevated negativity indicators  

---

## Project Architecture  

Each module is fully encapsulated to ensure **readability, scalability, and modularity**.

---

## Data Sources  
The dashboard uses datasets derived from **Yelp Open Dataset** and cleaned/filtered for MSBA 503 course use:

- `hair_sample_15k.parquet`  
- `chipotle_sample_15k.parquet`  
- `business_aggregated_sample.csv`  

Files include business metadata, review text, emotion tagging, and aggregated KPIs.

---

## Tech Stack  

### **Languages & Libraries**
- Python 3.11  
- Polars  
- Pandas  
- Streamlit  
- Statsmodels (ARIMA)  
- Matplotlib / Altair  
- NRC emotion lexicon  
- Scikit-learn utilities  

### **Deployment**
- **Streamlit Cloud** (recommended)  
- GitHub for full version control  

---

## How to Run Locally

### **1. Clone the repository**
```bash
git clone https://github.com/alexgsnyder/yelp-universal-dashboard.git
cd yelp-universal-dashboard
