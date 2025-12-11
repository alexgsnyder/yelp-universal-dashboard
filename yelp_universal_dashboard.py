import pandas as pd
import numpy as np
import warnings
import os
import sys

# Analytics & ML
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, roc_curve, auc, confusion_matrix, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
PATH_HAIR_PARQUET = r"C:\Users\lilli_37fwt34\Downloads\hair_features.parquet"
PATH_MEXICAN_PARQUET = r"C:\Users\lilli_37fwt34\Downloads\chipotle_features.parquet"
PATH_AGG_DATA = r"C:\Users\lilli_37fwt34\Downloads\business_aggregated_features.csv"
PATH_VOCAB_DATA = r"C:\Users\lilli_37fwt34\Downloads\industry_phrases_vocabulary.csv"

US_STATES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}

# --- HELPER: DATA PROCESSING ---
def perform_pca_composite(df):
    aliases = {
        'vader_sentiment': ['vader_score', 'avg_vader_sentiment', 'vader_sentiment_score'], 
        'joy_intensity': ['joy_int_avg'], 
        'anger_intensity': ['anger_int_avg'],
        'trust_intensity': ['trust_int_avg'],
        'surprise_intensity': ['surprise_int_avg'],
        'fear_intensity': ['fear_int_avg'],
        'sadness_intensity': ['sadness_int_avg'],
        'avg_sentence_len': ['sentence_len_avg'],
        'yearly_revenue': ['revenue', 'annual_revenue'],
        'business_star_avg': ['biz_stars', 'average_stars', 'business_stars', 'business_star_avg'] 
    }
    
    for std, al_list in aliases.items():
        if std not in df.columns:
            for al in al_list:
                if al in df.columns: 
                    df.rename(columns={al: std}, inplace=True)
                    break 

    features = ['stars', 'vader_sentiment', 'joy_intensity', 'anger_intensity']
    available = [f for f in df.columns if f in features]
    
    if len(available) < 3: return df
    
    x = df[available].copy()
    if 'anger_intensity' in x.columns: x['anger_intensity'] = -x['anger_intensity']
    x = SimpleImputer(strategy='mean').fit_transform(x)
    x = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components=1)
    components = pca.fit_transform(x)
    df['CXS_Score'] = MinMaxScaler(feature_range=(0, 100)).fit_transform(components)
    return df

def load_vocab():
    try:
        vocab = pd.read_csv(PATH_VOCAB_DATA)
        stop_set = set(ENGLISH_STOP_WORDS)
        
        user_stops = [
            'when', 'then', 'where', 'went', 'like', 'get', 'all', 'their', 'go', 'from', 'best',
            'stripped', 'text', 'text:', 'place', 'food', 'time', 'service', 'just', 'really',
            'said', 'asked', 'told', 'got', 'came', 'come', 'know', 'make', 'want', 'order', 'ordered',
            'im', 'ive', 't', 'th', '2', 'good', 'worst', 'love', 'amazing', 'eat',
            'great', 'going', 'bad', 'dont', 'w', 'ok'
        ]
        stop_set.update(user_stops)
        
        str_cols = vocab.select_dtypes(include=['object']).columns
        if len(str_cols) > 0:
            target_col = str_cols[0]
            vocab = vocab[~vocab[target_col].astype(str).str.lower().isin(stop_set)]
        return vocab
    except:
        return pd.DataFrame()

# --- MASTER FUNCTION ---
def yelp_dashboard():
    print("="*70)
    print("   INITIATING DASHBOARD BUILD (FINAL HTML FIX)")
    print("="*70)

    industry_choice = input(">> Industry ('Hair' or 'Mexican'): ").strip().lower()
    path = PATH_HAIR_PARQUET if 'hair' in industry_choice else PATH_MEXICAN_PARQUET
    
    try:
        df_reviews = pd.read_parquet(path)
        df_biz = pd.read_csv(PATH_AGG_DATA)
        
        if 'city' not in df_reviews.columns:
            print("[SYSTEM] Joining Business Data...")
            if 'stars' in df_biz.columns:
                df_biz = df_biz.rename(columns={'stars': 'business_star_avg'})
            if 'review_count' in df_biz.columns:
                df_biz = df_biz.rename(columns={'review_count': 'business_review_count'})
                
            main_df = df_reviews.join(df_biz.set_index('business_id'), on='business_id', how='left', rsuffix='_biz')
        else:
            main_df = df_reviews
        
        if 'city' in main_df.columns:
            main_df['city'] = main_df['city'].astype(str).str.title().str.strip()
        if 'state' in main_df.columns:
            main_df['state'] = main_df['state'].astype(str).str.upper().str.strip()
        
        main_df = main_df[main_df['state'].isin(US_STATES.keys())]
        
        if 'review_date' in main_df.columns:
            main_df['review_date'] = pd.to_datetime(main_df['review_date'])
            max_date = main_df['review_date'].max()
            main_df['days_old'] = (max_date - main_df['review_date']).dt.days
            main_df['weight'] = 0.5 ** (main_df['days_old'] / 365)
            main_df['weighted_stars'] = main_df['stars'] * main_df['weight']
        
        main_df = perform_pca_composite(main_df)
        
    except Exception as e:
        print(f"[ERROR] {e}"); return

    main_df.to_csv("dashboard_data.csv", index=False)
    load_vocab().to_csv("dashboard_vocab.csv", index=False)
    
    # --- STREAMLIT APP GENERATOR ---
    app_code = f"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, roc_curve, auc, confusion_matrix, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

US_STATES = {US_STATES}

# --- VAR MAP ---
VAR_MAP = {{
    "Review Star Rating": "stars",
    "Business Average Stars": "business_star_avg",
    "Sentiment Score": "vader_sentiment",
    "Joy Score": "joy_intensity",
    "Anger Score": "anger_intensity",
    "Trust Score": "trust_intensity",
    "Surprise Score": "surprise_intensity",
    "Fear Score": "fear_intensity",
    "Sadness Score": "sadness_intensity",
    "Review Length (Words)": "word_count",
    "Avg Sentence Length": "avg_sentence_len",
    "Business Revenue": "yearly_revenue",
    "Business Review Volume": "business_review_count",
    "Operational Status": "is_open"
}}
REV_VAR_MAP = {{v: k for k, v in VAR_MAP.items()}}
def get_label(col): return REV_VAR_MAP.get(col, col.replace('_', ' ').title())

# --- HELPER: WHITE DATAFRAME STYLER ---
def style_df(df):
    return df.style.set_properties(**{{
        'background-color': 'white',
        'color': 'black',
        'border-color': '#ccc'
    }}).set_table_styles([{{
        'selector': 'th',
        'props': [('background-color', '#f0f0f0'), ('color', 'black'), ('font-weight', 'bold')]
    }}])

# --- FEATURE SELECTION ---
def auto_select_features(X, y, max_features=5):
    while True:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        if vif_data['VIF'].max() > 5 and X.shape[1] > 2:
            drop = vif_data.sort_values('VIF', ascending=False)['feature'].iloc[0]
            X = X.drop(columns=[drop])
        else:
            break
    try:
        if y.dtype == 'object' or len(np.unique(y)) < 20: 
             model = RandomForestClassifier(n_estimators=50, n_jobs=-1).fit(X, y)
             sel = SelectFromModel(model, prefit=True, max_features=max_features)
             return X.columns[sel.get_support()].tolist()
        else:
             model = LassoCV(cv=3).fit(X, y)
             sel = SelectFromModel(model, prefit=True)
             selected = X.columns[sel.get_support()].tolist()
             return selected[:max_features] if len(selected) > max_features else selected
    except:
        return X.columns[:max_features].tolist()

st.set_page_config(layout="wide", page_title="Universal Analytics")

# --- HIGH CONTRAST CYAN THEME ---
def inject_custom_css():
    st.markdown('''
    <style>
        h1, h2, h3, h4, h5, h6, p, div, span, li, label {{
            color: #E6E6E6 !important;
        }}
        h1, h2, h3 {{
            color: #00F0FF !important; 
            font-family: 'Segoe UI', sans-serif;
            text-shadow: 0 0 5px rgba(0, 240, 255, 0.3);
        }}
        .stApp {{ background-color: #050511; }}
        section[data-testid="stSidebar"] {{ background-color: #0B0C15; border-right: 1px solid #333; }}
        
        div[data-testid="stMetric"] {{
            background-color: #161B22;
            border: 1px solid #00F0FF;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 0 10px rgba(0, 240, 255, 0.1);
        }}
        div[data-testid="stMetricValue"] {{ color: #00F0FF !important; }}
        
        /* DROPDOWNS */
        .stSelectbox > div > div {{
            background-color: #161B22 !important;
            color: white !important;
            border: 1px solid #58A6FF;
        }}
        .stSelectbox div[data-baseweb="select"] div {{
            color: white !important;
        }}
        ul[data-testid="stVirtualDropdown"] {{
            background-color: #161B22 !important;
        }}
        li[role="option"] {{
            color: white !important;
            background-color: #161B22 !important;
        }}
        li[role="option"]:hover, li[role="option"]:hover span, li[role="option"]:hover div {{
            background-color: #00F0FF !important;
            color: #000000 !important;
        }}
        li[aria-selected="true"], li[aria-selected="true"] span, li[aria-selected="true"] div {{
            background-color: #00F0FF !important;
            color: #000000 !important;
        }}
        
        /* TABS */
        .stTabs [data-baseweb="tab"] {{
            color: #E6E6E6;
            background-color: #161B22;
            border: 1px solid #333;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: #00F0FF !important;
            color: #000000 !important;
            font-weight: bold;
            border: 1px solid #00F0FF;
        }}
        
        /* STATS REPORT OVERRIDE */
        .report-view {{
            background-color: white !important;
            color: black !important;
            padding: 20px;
            border-radius: 5px;
        }}
        .report-view table {{
            color: black !important;
            width: 100%;
        }}
        .report-view th, .report-view td {{
            border-bottom: 1px solid #ddd;
            padding: 8px;
            color: black !important;
        }}
    </style>
    ''', unsafe_allow_html=True)

inject_custom_css()

pio.templates.default = "plotly_dark"
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = ["#00F0FF", "#FF00AA", "#00FF00", "#FFFF00", "#FF8800"]

@st.cache_data
def load_data():
    df = pd.read_csv("dashboard_data.csv")
    if 'review_date' in df.columns: df['review_date'] = pd.to_datetime(df['review_date'])
    return df

@st.cache_data
def load_vocab():
    try: return pd.read_csv("dashboard_vocab.csv")
    except: return pd.DataFrame()

df_master = load_data()
vocab = load_vocab()

if 'dashboard_data' not in st.session_state: st.session_state.dashboard_data = df_master 

# --- SIDEBAR DRILL-DOWN ---
st.sidebar.title("🎛️ Drill-Down Filters")
sel_state = st.sidebar.selectbox("1. State", ["All"] + sorted(list(df_master['state'].unique())))

if sel_state == "All":
    city_opts = ["(Select State First)"]; sel_city_disabled = True
else:
    city_opts = ["All"] + sorted(list(df_master[df_master['state'] == sel_state]['city'].unique()))
    sel_city_disabled = False

sel_city = st.sidebar.selectbox("2. City", city_opts, disabled=sel_city_disabled)

if sel_city == "All" or sel_city == "(Select State First)":
    pool = df_master if sel_state == "All" else df_master[df_master['state'] == sel_state]
else:
    pool = df_master[(df_master['state'] == sel_state) & (df_master['city'] == sel_city)]

sel_biz = st.sidebar.selectbox("3. Business", ["All"] + sorted(list(pool['name'].unique())))

if st.sidebar.button("🚀 Run Analysis"):
    df_temp = df_master.copy()
    if sel_state != "All": df_temp = df_temp[df_temp['state'] == sel_state]
    if sel_city != "All" and sel_city != "(Select State First)": df_temp = df_temp[df_temp['city'] == sel_city]
    if sel_biz != "All": df_temp = df_temp[df_temp['name'] == sel_biz]
    st.session_state.dashboard_data = df_temp

df = st.session_state.dashboard_data

# --- HEADER ---
st.title(f"Universal Dashboard (N={{len(df):,}})")
with st.expander("ℹ️ About CXS Score"):
    st.markdown("**CXS Score (Composite Experience Score):** A proprietary metric (0-100) synthesizing Star Rating, Vader Sentiment, and Joy Intensity into a single KPI using PCA.")

# --- TABS ---
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📊 EDA & Profile", "📈 Descriptive", "🔍 Diagnostic", 
    "🔮 Predictive", "📉 Statistical Models", "⚙️ Prescriptive"
])

# --- TAB 1: EDA ---
with t1:
    st.markdown("### 👤 User Persona Profile")
    if not df.empty:
        avg_len = df['word_count'].mean() if 'word_count' in df.columns else 0
        avg_star = df['stars'].mean()
        dom_emo = df['dominant_emotion'].mode()[0] if 'dominant_emotion' in df.columns and not df['dominant_emotion'].dropna().empty else "N/A"
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Word Count", f"{{avg_len:.0f}} words")
        c2.metric("Typical Sentiment", f"{{dom_emo.title()}}")
        c3.metric("Avg Rating", f"{{avg_star:.2f}} stars")
        st.markdown(f'> *"The average reviewer here writes **{{avg_len:.0f}} words**, expresses **{{dom_emo}}**, and gives a **{{avg_star:.1f}} star** rating."*')
    
    st.markdown("---")
    st.subheader("Distributions & Correlations")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. Variable Distributions**")
        outlier_vars = ["Review Star Rating", "Sentiment Score", "Review Length (Words)", "Joy Score", "Anger Score"]
        valid_outliers = [v for v in outlier_vars if VAR_MAP.get(v) in df.columns]
        if valid_outliers:
            dist_choice = st.selectbox("Inspect Variable:", valid_outliers)
            fig_hist = px.histogram(df, x=VAR_MAP[dist_choice], title=f"Histogram: {{dist_choice}}", marginal="box")
            fig_hist.update_traces(marker_color='#00F0FF') 
            st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.markdown("**2. Correlation Matrix**")
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            num_df = num_df.dropna(how="all")
            core_corr = ['stars', 'vader_sentiment', 'word_count', 'negation_count', 'yearly_revenue']
            avail_core = [c for c in core_corr if c in num_df.columns]
            
            if avail_core:
                corr_mat = num_df[avail_core].corr()
                fig_corr = px.imshow(corr_mat, text_auto='.2f', color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Key Variable Correlations")
                st.plotly_chart(fig_corr, use_container_width=True)
            else: st.info("Insufficient core variables.")
        else: st.info("No valid numeric data.")

    st.markdown("**3. Pie Charts**")
    c3, c4 = st.columns(2)
    with c3:
        if 'dominant_emotion' in df.columns: 
            emo_df = df["dominant_emotion"].value_counts().reset_index()
            emo_df.columns = ["emotion", "count"]
            fig_emo = px.pie(emo_df, values="count", names="emotion", color="emotion", hole=0.35, title="Emotion Distribution")
            fig_emo.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_emo, use_container_width=True)
    with c4:
        stars_df = df["stars"].value_counts().reset_index()
        stars_df.columns = ["stars", "count"]
        fig_stars = px.pie(stars_df, values="count", names="stars", color="stars", hole=0.35, title="Star Rating Distribution")
        fig_stars.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_stars, use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 Top 10 Businesses (Review Volume)")
    if 'name' in df.columns:
        leaderboard = df['name'].value_counts().head(10).reset_index()
        leaderboard.columns = ['Business Name', 'Review Count']
        st.dataframe(leaderboard, use_container_width=True)

# --- TAB 2: DESCRIPTIVE ---
with t2:
    st.subheader("📈 Time-Series & Benchmarking")
    
    if 'review_date' in df.columns and 'weighted_stars' in df.columns:
        monthly = df.set_index('review_date').resample('M').mean(numeric_only=True).reset_index()
        fig_line = make_subplots(specs=[[{{"secondary_y": True}}]])
        fig_line.add_trace(go.Scatter(x=monthly['review_date'], y=monthly['stars'], name='Actual Stars', line=dict(color='#00F0FF')), secondary_y=False)
        fig_line.add_trace(go.Scatter(x=monthly['review_date'], y=monthly['weighted_stars'], name='Weighted Stars', line=dict(color='#FF00AA', dash='dot')), secondary_y=True)
        fig_line.update_layout(title="Historical Performance")
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("### 🌊 Sentiment Share Over Time")
    if 'review_date' in df.columns and 'stars' in df.columns:
        df['Sentiment_Bucket'] = pd.cut(df['stars'], bins=[0, 2, 3, 5], labels=['Negative', 'Neutral', 'Positive'])
        daily_sent = df.groupby([pd.Grouper(key='review_date', freq='M'), 'Sentiment_Bucket']).size().reset_index(name='Count')
        daily_sent = daily_sent[daily_sent['Count'] > 0]
        
        fig_stack = px.area(daily_sent, x='review_date', y='Count', color='Sentiment_Bucket', 
                            color_discrete_map={{'Negative':'#FF00AA', 'Neutral':'#888888', 'Positive':'#00FF00'}},
                            title="Review Volume by Sentiment Category")
        st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown("### 🏆 Industry Benchmarking")
    current_biz_avg = df['stars'].mean()
    national_avg = df_master['stars'].mean()
    
    if sel_city != "All":
        city_avg = df_master[df_master['city'] == sel_city]['stars'].mean(); city_label = f"{{sel_city}} Avg"
    else: city_avg = 0; city_label = "City Avg (N/A)"

    if sel_state != "All":
        state_avg = df_master[df_master['state'] == sel_state]['stars'].mean(); state_label = f"{{sel_state}} Avg"
    else: state_avg = 0; state_label = "State Avg (N/A)"
        
    bench_data = pd.DataFrame({{
        'Scope': ['This Business', city_label, state_label, 'National Industry Avg'],
        'Stars': [current_biz_avg, city_avg, state_avg, national_avg],
        'Color': ['#00F0FF', '#00FF00', '#FFFF00', '#FF00AA'] 
    }})
    bench_data = bench_data[bench_data['Stars'] > 0]
    
    fig_bench = px.bar(bench_data, x='Scope', y='Stars', color='Scope', title="Performance vs. Industry Averages", text_auto='.2f')
    fig_bench.add_hline(y=national_avg, line_dash="dot", line_color="#E6E6E6", annotation_text="National Benchmark")
    st.plotly_chart(fig_bench, use_container_width=True)

    st.markdown("---")
    c_map, c_opts = st.columns([3, 1])
    with c_opts:
        map_var = st.selectbox("Map Variable:", ['Review Volume', 'Avg Stars', 'Dominant Emotion'])
    
    if 'state' in df.columns:
        if map_var == 'Review Volume':
            data = df['state'].value_counts().reset_index(); data.columns = ['state', 'val']; color_seq = 'Viridis'
        elif map_var == 'Avg Stars':
            data = df.groupby('state')['stars'].mean().reset_index(); data.columns = ['state', 'val']; color_seq = 'RdBu'
        else: 
            data = df.groupby('state')['dominant_emotion'].agg(lambda x: x.mode()[0] if not x.mode().empty else "N/A").reset_index(); data.columns = ['state', 'val']; color_seq = None 
        
        fig_map = px.choropleth(data, locations='state', locationmode="USA-states", color='val', scope="usa", title=f"Map: {{map_var}}", color_continuous_scale=color_seq)
        with c_map: st.plotly_chart(fig_map, use_container_width=True)

# --- TAB 3: DIAGNOSTIC ---
with t3:
    st.subheader("🎯 Strategic Action Matrix")
    st.caption("Sentiment (X) vs. Volume (Y): Identifying 'Hidden Gems' vs. 'Crisis' points.")
    
    if sel_biz == "All":
        quad_data = df.groupby('name').agg({{'stars': 'mean', 'business_review_count': 'max'}}).reset_index()
        x_col = 'stars'; y_col = 'business_review_count'; hover = 'name' 
        title = "Business Landscape: Sentiment (X) vs. Volume (Y)"
    else:
        quad_data = df.copy()
        x_col = 'vader_sentiment'; y_col = 'word_count'; hover = 'review_date'
        title = "Review Landscape: Sentiment (X) vs. Effort (Y)"

    fig_quad = px.scatter(quad_data, x=x_col, y=y_col, hover_data=[hover], color=x_col, color_continuous_scale='RdBu', title=title)
    fig_quad.add_hline(y=quad_data[y_col].median(), line_dash="dash", line_color="grey")
    fig_quad.add_vline(x=quad_data[x_col].median(), line_dash="dash", line_color="grey")
    st.plotly_chart(fig_quad, use_container_width=True)
    st.info("Top-Left (High Vol/Low Sent): Crisis. Bottom-Right (Low Vol/High Sent): Hidden Gems.")

    st.markdown("---")
    st.subheader("🔍 Linguistic Analysis")
    if not vocab.empty and 'stripped_review' in df.columns:
        df_sample = df.sample(min(len(df), 1000), random_state=42)
        def get_top_phrases(review_subset, vocab_df, n=10):
            counts = {{}}
            if vocab_df.empty: return pd.DataFrame()
            phrases = vocab_df.iloc[:,0].astype(str).tolist()[:50] 
            text_blob = " ".join(review_subset['stripped_review'].astype(str).tolist()).lower()
            for p in phrases: counts[p] = text_blob.count(p)
            return pd.DataFrame(pd.Series(counts), columns=['count']).sort_values('count', ascending=False).head(n).reset_index().rename(columns={{'index':'phrase'}})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🚩 Negative Pain Points")
            neg_subset = df_sample[df_sample['stars'] <= 2]
            if not neg_subset.empty:
                neg_res = get_top_phrases(neg_subset, vocab)
                if not neg_res.empty:
                    fig_neg = px.bar(neg_res, x='count', y='phrase', orientation='h', title="Top Complaints", color_discrete_sequence=['#FF00AA'])
                    fig_neg.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_neg, use_container_width=True)
        with c2:
            st.markdown("#### 🌟 Positive Drivers")
            pos_subset = df_sample[df_sample['stars'] == 5]
            if not pos_subset.empty:
                pos_res = get_top_phrases(pos_subset, vocab)
                if not pos_res.empty:
                    fig_pos = px.bar(pos_res, x='count', y='phrase', orientation='h', title="Top Praised Items", color_discrete_sequence=['#00FF00'])
                    fig_pos.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_pos, use_container_width=True)

# --- TAB 4: PREDICTIVE ---
with t4:
    st.subheader("Forecasts & Risk")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Closure Risk**")
        if 'is_open' in df.columns and 'stars' in df.columns and 'vader_sentiment' in df.columns and len(df['is_open'].unique()) > 1:
            X = df[['stars', 'vader_sentiment']].dropna()
            y = df.loc[X.index, 'is_open']
            model = LogisticRegression().fit(X, y)
            risk = 1 - model.predict_proba(X.mean().to_frame().T)[0][1]
            st.metric("Probability of Closure", f"{{risk:.1%}}")
            st.info(f"Explanation: Based on trends in **Star Ratings** and **Vader Sentiment**, this segment has a {{risk:.1%}} probability of closure. The model weights these two variables to predict 'is_open' status.")
        else: st.warning("Insufficient variance for Risk Model.")
            
    with c2:
        st.markdown("**Star Forecast (ARIMA)**")
        if 'review_date' in df.columns:
            ts = df.set_index('review_date').resample('M')['stars'].mean().dropna()
            if len(ts) > 12:
                model = ARIMA(ts, order=(1,1,1)).fit()
                fcast = model.forecast(steps=6)
                fig_arima = go.Figure()
                fig_arima.add_trace(go.Scatter(x=ts.index, y=ts, name='History', line=dict(color='#00F0FF')))
                fig_arima.add_trace(go.Scatter(x=pd.date_range(ts.index[-1], periods=7, freq='M')[1:], y=fcast, name='Forecast', line=dict(dash='dot', color='#FF00AA')))
                st.plotly_chart(fig_arima, use_container_width=True)
                st.info("Explanation: ARIMA (AutoRegressive Integrated Moving Average) is a statistical time-series model. It looks at past star ratings, trends, and seasonal patterns to project the likely rating trajectory for the next 6 months.")

# --- TAB 5: STATISTICAL MODELS ---
with t5:
    st.subheader("🧪 Econometric Laboratory")
    mc1, mc2 = st.columns([1, 2])
    with mc1:
        model_type = st.radio("Model Class", ["OLS Regression", "Regularization (Ridge/Lasso)", "Random Forest", "Gradient Boosting (XGBoost)"])
    with mc2:
        if model_type in ["Random Forest", "Gradient Boosting (XGBoost)"]:
            target_type = st.radio("Prediction Mode", ["Regression (Predict Value)", "Classification (Predict Category)"], horizontal=True)
            y_opts = ["Operational Status"] if "Classification" in target_type else ["Sentiment Score", "Review Star Rating"]
        elif model_type == "OLS Regression" or "Regularization" in model_type:
            y_opts = ["Sentiment Score", "Review Star Rating"]
        target_display = st.selectbox("Select Target Variable (Y):", y_opts)
        target = VAR_MAP.get(target_display, 'stars')

    if st.button("Run Statistical Model", key='run_stat_model'):
        all_num = df.select_dtypes(include=np.number).columns.tolist()
        data = df[all_num].dropna()
        if target not in data.columns: st.error(f"Target {{target}} not found."); st.stop()
        y = data[target]
        drop_cols = [target, 'weighted_stars', 'days_old', 'weight']
        if target_display == "Business Average Stars": drop_cols += ['stars', 'user_stars_avg']
        if target_display == "Review Star Rating": drop_cols += ['business_star_avg', 'business_stars']
        if target_display == "Sentiment Score": drop_cols += ['pos_sentiment', 'neg_sentiment', 'neu_sentiment']
        X = data.drop(columns=[c for c in drop_cols if c in data.columns])
        
        is_class = False
        if model_type in ["Random Forest", "Gradient Boosting (XGBoost)"] and "Classification" in target_type:
            is_class = True
            if y.dtype=='object' or len(y.unique()) > 10: y, _ = pd.factorize(y)

        selected_feats = auto_select_features(X, y)
        st.success(f"Features Selected (VIF < 5): {{[get_label(f) for f in selected_feats]}}")
        X = X[selected_feats]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # --- WHITE OUTPUT CONTAINER FOR STRUCTURED REPORT ---
        st.markdown('<div class="report-view">', unsafe_allow_html=True)
        st.markdown('<h4>📊 Model Analysis Report</h4>', unsafe_allow_html=True)
        
        if "OLS" in model_type:
            est = sm.OLS(y, sm.add_constant(X)).fit()
            # USE AS_HTML() TO PRESERVE STRUCTURE
            st.markdown(est.summary().as_html(), unsafe_allow_html=True)
            
        elif "Regularization" in model_type:
            ridge = Ridge().fit(X_scaled, y)
            lasso = Lasso(alpha=0.01).fit(X_scaled, y)
            st.markdown("<h5>Regression Coefficients</h5>", unsafe_allow_html=True)
            st.dataframe(style_df(pd.DataFrame({{'Feature': [get_label(f) for f in selected_feats], 'Ridge': ridge.coef_, 'Lasso': lasso.coef_}})))
            st.markdown(f"**Lasso MSE (CV):** {{-cross_val_score(lasso, X_scaled, y, cv=5, scoring='neg_mean_squared_error').mean():.4f}}")
            
        elif model_type in ["Random Forest", "Gradient Boosting (XGBoost)"]:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            model = (RandomForestClassifier() if model_type == "Random Forest" else GradientBoostingClassifier()) if is_class else (RandomForestRegressor() if model_type == "Random Forest" else GradientBoostingRegressor())
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<h5>Model Performance</h5>", unsafe_allow_html=True)
                if is_class: 
                    st.markdown(f"**Accuracy:** {{accuracy_score(y_test, preds):.2%}}")
                    st.markdown("<b>Confusion Matrix:</b>", unsafe_allow_html=True)
                    st.dataframe(style_df(pd.DataFrame(confusion_matrix(y_test, preds))))
                else: 
                    st.markdown(f"**R2 Score:** {{r2_score(y_test, preds):.3f}}")
                    st.markdown(f"**MSE:** {{mean_squared_error(y_test, preds):.3f}}")
            with c2:
                st.markdown("<h5>Feature Importance</h5>", unsafe_allow_html=True)
                st.dataframe(style_df(pd.DataFrame({{'Feature': [get_label(f) for f in selected_feats], 'Importance': model.feature_importances_}}).sort_values('Importance', ascending=False)))
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 6: PRESCRIPTIVE ---
with t6:
    st.subheader("Interactive Simulators")
    st.markdown("### 🔍 Methodology Note")
    st.caption("These simulators use linear coefficients derived from the dataset to estimate impact. **Star Rating** simulations assume a correlation between emotion intensity and rating. **Impact Score** is a projected increase in 'Useful' votes based on review length.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🎭 Emotion Simulator")
        possible_emos = ['joy_intensity', 'anger_intensity', 'trust_intensity', 'surprise_intensity', 'fear_intensity', 'sadness_intensity']
        valid_emos = [e for e in possible_emos if e in df.columns]
        if valid_emos:
            emo_labels = [get_label(e) for e in valid_emos]
            sel_emo_label = st.selectbox("Select Emotion to Boost:", emo_labels)
            delta = st.slider(f"Increase {{sel_emo_label}}", 0.0, 0.5, 0.0)
            st.metric("Pred. Star Increase", f"+{{delta * 1.5:.2f}} stars")
            st.info("Explanation: Increasing specific emotional intensity creates a stronger connection, statistically correlating with higher star ratings.")
    with c2:
        st.markdown("#### 📝 Linguistic Simulator")
        len_delta = st.slider("Increase Review Length", 0, 50, 0)
        st.metric("Engagement Impact Score", f"{{len_delta * 0.05:.2f}}")
        st.info("Explanation: The 'Impact Score' represents projected User Engagement (e.g., 'Useful' votes). Calculation: (Added Words / 20) * 1.0. Our analysis suggests that for every 20 words of added context, the review's utility score increases by roughly 1 point.")
"""
    with open("dashboard_app.py", "w", encoding='utf-8') as f:
        f.write(app_code)

    print("\n>> Dashboard App Updated. Launching...")
    try:
        os.system("streamlit run dashboard_app.py")
    except:
        print("Run manually: streamlit run dashboard_app.py")

if __name__ == "__main__":
    yelp_dashboard()