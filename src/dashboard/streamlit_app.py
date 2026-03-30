"""
streamlit_app.py — Interactive Streamlit Dashboard for Flight Delay Prediction System
"""

import sys
import json
import sqlite3
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from pathlib import Path
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    AIRLINES, AIRPORTS, WEATHER_CODES,
    METRICS_PATH, FEATURE_IMP_PLOT, CONFUSION_PLOT, ROC_PLOT,
    SQLITE_PATH, PROCESSED_CSV, RAW_FLIGHTS_CSV
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Flight Delay Prediction System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { background: #0D1117; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
    border: 1px solid #2A2A4E;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value { font-size: 2.2em; font-weight: 700; color: #4FC3F7; }
.metric-label { font-size: 0.85em; color: #9E9E9E; margin-top: 4px; }

/* Prediction result */
.pred-delayed {
    background: linear-gradient(135deg, #FF1744 0%, #D32F2F 100%);
    border-radius: 12px; padding: 24px; text-align: center;
    box-shadow: 0 4px 24px rgba(255,23,68,0.3);
    animation: pulse 2s infinite;
}
.pred-ontime {
    background: linear-gradient(135deg, #00E676 0%, #00897B 100%);
    border-radius: 12px; padding: 24px; text-align: center;
    box-shadow: 0 4px 24px rgba(0,230,118,0.3);
}
.pred-title { font-size: 1.8em; font-weight: 700; color: white; }
.pred-subtitle { font-size: 1.1em; color: rgba(255,255,255,0.85); margin-top: 6px; }

@keyframes pulse {
    0%, 100% { box-shadow: 0 4px 24px rgba(255,23,68,0.3); }
    50%       { box-shadow: 0 4px 36px rgba(255,23,68,0.6); }
}

/* Gauge label */
.gauge-label {
    font-size: 1.4em; font-weight: 600;
    text-align: center; padding: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 0.9em; font-weight: 600;
    color: #9E9E9E; padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    color: #4FC3F7 !important;
    border-bottom: 2px solid #4FC3F7 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0D1117;
    border-right: 1px solid #1F2937;
}

/* Divider */
hr { border-color: #1F2937; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    try:
        from src.models.predictor import get_predictor
        return get_predictor(), None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=60)
def load_raw_data():
    try:
        return pd.read_csv(RAW_FLIGHTS_CSV, parse_dates=["scheduled_dep"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_metrics():
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def gauge_chart(prob: float) -> go.Figure:
    color = "#FF1744" if prob >= 0.70 else "#FF9800" if prob >= 0.40 else "#00E676"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        title={"text": "Delay Probability (%)", "font": {"color": "white", "size": 16}},
        number={"suffix": "%", "font": {"color": color, "size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9E9E9E",
                     "tickfont": {"color": "#9E9E9E"}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#1A1A2E",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "#1A2E1A"},
                {"range": [40, 70], "color": "#2E2A1A"},
                {"range": [70, 100],"color": "#2E1A1A"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.9,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=40, b=0, l=20, r=20),
        paper_bgcolor="#0D1117",
        font={"color": "white"},
    )
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-size:2.5em;">✈️</div>
        <div style="font-size:1.2em; font-weight:700; color:#4FC3F7;">Flight Delay</div>
        <div style="font-size:0.85em; color:#9E9E9E;">Prediction System</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🛫 Flight Details")
    airline       = st.selectbox("Airline",     AIRLINES, index=0)
    origin        = st.selectbox("Origin",       AIRPORTS, index=0)
    destination   = st.selectbox("Destination", [a for a in AIRPORTS if a != origin], index=1)
    dep_date      = st.date_input("Departure Date")
    dep_time      = st.time_input("Departure Time")
    distance_km   = st.slider("Distance (km)", 200, 3200, 1100, step=50)

    st.markdown("### 🌤️ Weather Conditions")
    temperature_c    = st.slider("Temperature (°C)", -5, 48, 28)
    wind_speed_kmh   = st.slider("Wind Speed (km/h)", 0, 120, 15)
    visibility_km    = st.slider("Visibility (km)", 0.5, 15.0, 10.0, step=0.5)
    precipitation_mm = st.slider("Precipitation (mm)", 0.0, 80.0, 0.0, step=0.5)
    weather_code     = st.selectbox(
        "Weather Condition",
        options=list(WEATHER_CODES.keys()),
        format_func=lambda x: f"{x} — {WEATHER_CODES[x]}"
    )

    st.markdown("### 🏙️ Airport Traffic")
    congestion_index = st.slider("Congestion Index", 0.0, 1.0, 0.40, step=0.01)

    st.divider()
    predict_btn = st.button("🔮 Predict Delay", use_container_width=True, type="primary")


# ─── Main Layout ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 20px 0 10px 0;">
    <h1 style="font-size:2em; font-weight:700; color:#4FC3F7; margin-bottom:4px;">
        ✈️ Flight Delay Prediction System
    </h1>
    <p style="color:#9E9E9E; font-size:0.95em;">
        ML-powered predictions using Random Forest, XGBoost & Logistic Regression
        trained on 100,000+ synthetic flight records.
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Prediction", "📊 Analytics Dashboard",
    "📈 Model Performance",  "🗄️ Database Viewer"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    predictor, err = load_predictor()

    if err:
        st.error(f"⚠️ Model not loaded: {err}\n\n"
                 f"Run `python run_pipeline.py` to train the model first.")
    else:
        if predict_btn or "last_result" in st.session_state:
            if predict_btn:
                inp = {
                    "airline":         airline,
                    "origin":          origin,
                    "destination":     destination,
                    "scheduled_dep":   f"{dep_date}T{dep_time}",
                    "distance_km":     distance_km,
                    "temperature_c":   temperature_c,
                    "wind_speed_kmh":  wind_speed_kmh,
                    "visibility_km":   visibility_km,
                    "precipitation_mm":precipitation_mm,
                    "weather_code":    weather_code,
                    "congestion_index":congestion_index,
                }
                with st.spinner("Running model inference..."):
                    result = predictor.predict(inp)
                st.session_state["last_result"] = result

            result = st.session_state["last_result"]
            prob   = result["delay_prob"]

            col1, col2 = st.columns([1.2, 1])

            with col1:
                # Result card
                css_class = "pred-delayed" if result["is_delayed"] else "pred-ontime"
                label     = "🚨 DELAYED" if result["is_delayed"] else "✅ ON TIME"
                st.markdown(f"""
                <div class="{css_class}">
                    <div class="pred-title">{label}</div>
                    <div class="pred-subtitle">{result['risk_level']}</div>
                    <div style="font-size:1em; color:rgba(255,255,255,0.8); margin-top:12px;">
                        Est. delay: <strong>{result['estimated_delay_min']} min</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Key metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Delay Probability", f"{result['delay_prob_pct']}%")
                c2.metric("Est. Delay",         f"{result['estimated_delay_min']} min")
                c3.metric("Model",               result["model_name"])

                # Input summary
                st.markdown("##### 📋 Input Summary")
                summ = result["input_summary"]
                df_summ = pd.DataFrame({
                    "Field": ["Route", "Dep Time", "Weather", "Congestion"],
                    "Value": [summ["route"], summ["dep_time"],
                              summ["weather"], summ["congestion"]]
                })
                st.dataframe(df_summ, hide_index=True, use_container_width=True)

            with col2:
                st.plotly_chart(gauge_chart(prob), use_container_width=True)

                # Risk breakdown
                st.markdown("##### ⚡ Key Risk Factors")
                factors = {
                    "☔ Precipitation":  min(precipitation_mm / 80, 1.0),
                    "💨 Wind Speed":     min(wind_speed_kmh / 120, 1.0),
                    "🌫️ Low Visibility": max(0, 1 - visibility_km / 15),
                    "🏙️ Congestion":    congestion_index,
                }
                for factor, val in factors.items():
                    col_f1, col_f2 = st.columns([2, 1])
                    col_f1.markdown(f"<small>{factor}</small>", unsafe_allow_html=True)
                    col_f2.progress(float(val))
        else:
            st.info("👈 Configure flight parameters in the sidebar and click **Predict Delay**")
            st.markdown("### How it works")
            c1, c2, c3 = st.columns(3)
            c1.markdown("""
            **🛠️ Pipeline**
            - Synthetic data (100K records)
            - Cleaning & imputation
            - Feature engineering
            """)
            c2.markdown("""
            **🤖 Models**
            - Logistic Regression
            - Random Forest
            - XGBoost
            """)
            c3.markdown("""
            **📊 Features**
            - Temporal encoding
            - Weather severity
            - Airport congestion
            """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    df_raw = load_raw_data()
    if df_raw.empty:
        st.warning("No data found. Run `python run_pipeline.py` first.")
    else:
        df_raw["dep_hour"]  = df_raw["scheduled_dep"].dt.hour
        df_raw["dep_month"] = df_raw["scheduled_dep"].dt.month
        df_raw["dep_dow"]   = df_raw["scheduled_dep"].dt.day_name()
        df_raw["weather_label2"] = df_raw.get("weather_label", df_raw.get("weather_code", "Unknown"))

        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        total   = len(df_raw)
        delayed = df_raw["is_delayed"].sum()
        c1.metric("Total Flights",    f"{total:,}")
        c2.metric("Delayed Flights",  f"{int(delayed):,}")
        c3.metric("Delay Rate",       f"{delayed/total:.1%}")
        c4.metric("Avg Delay (min)",  f"{df_raw[df_raw['is_delayed']==1]['delay_minutes'].mean():.1f}")

        st.divider()
        col_l, col_r = st.columns(2)

        # Delay by Airline
        with col_l:
            by_airline = (
                df_raw.groupby("airline")
                      .agg(total=("is_delayed","count"),
                           delayed=("is_delayed","sum"))
                      .assign(rate=lambda x: x["delayed"]/x["total"]*100)
                      .reset_index()
                      .sort_values("rate", ascending=False)
            )
            fig = px.bar(by_airline, x="airline", y="rate",
                         title="Delay Rate by Airline (%)",
                         color="rate",
                         color_continuous_scale="RdYlGn_r",
                         labels={"airline":"Airline","rate":"Delay Rate (%)"})
            fig.update_layout(
                paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                font_color="white", showlegend=False,
                title_font_color="#4FC3F7"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Delay by Hour
        with col_r:
            by_hour = (
                df_raw.groupby("dep_hour")
                      .agg(total=("is_delayed","count"),
                           delayed=("is_delayed","sum"))
                      .assign(rate=lambda x: x["delayed"]/x["total"]*100)
                      .reset_index()
            )
            fig2 = px.area(by_hour, x="dep_hour", y="rate",
                           title="Delay Rate by Departure Hour",
                           labels={"dep_hour":"Hour","rate":"Delay Rate (%)"},
                           color_discrete_sequence=["#4FC3F7"])
            fig2.update_layout(
                paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                font_color="white", title_font_color="#4FC3F7"
            )
            st.plotly_chart(fig2, use_container_width=True)

        col_l2, col_r2 = st.columns(2)

        # Delay distribution by weather
        with col_l2:
            fig3 = px.box(
                df_raw[df_raw["is_delayed"] == 1],
                x="weather_label2", y="delay_minutes",
                title="Delay Minutes by Weather Condition",
                color="weather_label2",
                labels={"weather_label2":"Weather","delay_minutes":"Delay (min)"}
            )
            fig3.update_layout(
                paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                font_color="white", title_font_color="#4FC3F7",
                showlegend=False,
                xaxis_tickangle=-30
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Monthly trend
        with col_r2:
            by_month = (
                df_raw.groupby("dep_month")
                      .agg(total=("is_delayed","count"),
                           delayed=("is_delayed","sum"))
                      .assign(rate=lambda x: x["delayed"]/x["total"]*100)
                      .reset_index()
            )
            month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                           7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            by_month["month_name"] = by_month["dep_month"].map(month_names)
            fig4 = px.line(by_month, x="month_name", y="rate",
                           title="Monthly Delay Rate Trend",
                           markers=True,
                           labels={"month_name":"Month","rate":"Delay Rate (%)"},
                           color_discrete_sequence=["#FF6B6B"])
            fig4.update_layout(
                paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                font_color="white", title_font_color="#4FC3F7"
            )
            st.plotly_chart(fig4, use_container_width=True)

        # Scatter: congestion vs delay  
        st.markdown("#### 🔗 Congestion Index vs Delay Minutes")
        sample = df_raw[df_raw["is_delayed"] == 1].sample(min(3000, len(df_raw)))
        fig5 = px.scatter(
            sample, x="congestion_index", y="delay_minutes",
            color="airline", opacity=0.5, size_max=8,
            labels={"congestion_index":"Congestion Index","delay_minutes":"Delay (min)"},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig5.update_layout(
            paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
            font_color="white", height=380
        )
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    metrics = load_metrics()
    if not metrics:
        st.warning("No metrics found. Run `python run_pipeline.py` first.")
    else:
        meta   = metrics.get("_meta", {})
        models = {k: v for k, v in metrics.items() if not k.startswith("_")}
        best   = meta.get("best_model", "")

        st.markdown(f"**Best Model:** `{best}` &nbsp;|&nbsp; "
                    f"**Samples:** `{meta.get('n_samples', 'N/A'):,}` &nbsp;|&nbsp; "
                    f"**Delay Rate:** `{meta.get('delay_rate', 0):.2%}`")
        st.divider()

        # Model comparison table
        rows = []
        for name, m in models.items():
            rows.append({
                "Model":     ("⭐ " if name == best else "") + name,
                "Accuracy":  f"{m.get('accuracy',0):.4f}",
                "Precision": f"{m.get('precision',0):.4f}",
                "Recall":    f"{m.get('recall',0):.4f}",
                "F1 Score":  f"{m.get('f1',0):.4f}",
                "ROC-AUC":   f"{m.get('roc_auc',0):.4f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # Bar comparison
        metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        comp_data = []
        for name, m in models.items():
            for metric in metric_names:
                comp_data.append({"Model": name, "Metric": metric.upper().replace("_","-"),
                                  "Score": m.get(metric, 0)})
        fig_comp = px.bar(
            pd.DataFrame(comp_data),
            x="Metric", y="Score", color="Model", barmode="group",
            title="Cross-Validated Model Comparison",
            color_discrete_sequence=["#4FC3F7", "#FF6B6B", "#69F0AE"]
        )
        fig_comp.update_layout(
            paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
            font_color="white", title_font_color="#4FC3F7",
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Saved plots
        c_conf, c_roc, c_imp = st.columns(3)
        for col, path, title in [
            (c_conf, CONFUSION_PLOT, "Confusion Matrix"),
            (c_roc,  ROC_PLOT,       "ROC Curve"),
            (c_imp,  FEATURE_IMP_PLOT,"Feature Importance"),
        ]:
            with col:
                if Path(path).exists():
                    img = Image.open(path)
                    st.image(img, caption=title, use_container_width=True)
                else:
                    st.info(f"{title} not found. Run evaluator.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Database Viewer
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🗄️ Live Database Query Viewer")
    if not SQLITE_PATH.exists():
        st.warning("Database not found. Run `python run_pipeline.py` first.")
    else:
        conn = sqlite3.connect(SQLITE_PATH)

        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()

        if tables:
            selected_table = st.selectbox("Select Table", tables)
            limit = st.slider("Row limit", 10, 1000, 100, step=10)
            try:
                df_tbl = pd.read_sql(
                    f"SELECT * FROM {selected_table} LIMIT {limit}", conn
                )
                st.markdown(f"**{len(df_tbl):,} rows** from `{selected_table}`")
                st.dataframe(df_tbl, use_container_width=True)

                # Row count per table
                st.markdown("### 📦 Table Sizes")
                size_rows = []
                for t in tables:
                    n = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {t}", conn).iloc[0,0]
                    size_rows.append({"Table": t, "Row Count": n})
                st.dataframe(pd.DataFrame(size_rows), hide_index=True)

            except Exception as e:
                st.error(f"Query error: {e}")

        # Custom SQL
        st.markdown("### 📝 Custom SQL Query")
        custom_sql = st.text_area(
            "Enter SQL",
            value="SELECT airline, COUNT(*) as total, SUM(is_delayed) as delayed\nFROM flights_raw\nGROUP BY airline\nORDER BY delayed DESC;",
            height=100
        )
        if st.button("▶ Run Query"):
            try:
                result_df = pd.read_sql(custom_sql, conn)
                st.dataframe(result_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

        conn.close()
