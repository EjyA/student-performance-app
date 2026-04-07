import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json

st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=DM+Serif+Display:ital@0;1&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main { background-color: #f7f4f0; }

    [data-testid="stSidebar"] {
        background-color: #1c2f45;
        border-right: none;
    }
    [data-testid="stSidebar"] * {
        color: #dce8f5 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 14.5px;
        font-weight: 400;
        letter-spacing: 0.01em;
        opacity: 0.88;
        padding: 4px 0;
    }

    h1 {
        font-family: 'DM Serif Display', serif !important;
        color: #1c2f45 !important;
        font-weight: 400 !important;
        font-size: 2.2rem !important;
        letter-spacing: -0.02em !important;
        line-height: 1.2 !important;
    }
    h2, h3 {
        font-family: 'DM Sans', sans-serif !important;
        color: #2a4060 !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 18px;
        padding: 22px 20px 18px;
        border: 1px solid rgba(28,47,69,0.08);
        box-shadow: 0 2px 12px rgba(28,47,69,0.06);
    }
    [data-testid="stMetricLabel"] {
        font-size: 11px;
        color: #8a9ab0;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        font-weight: 400;
        color: #1c2f45;
        letter-spacing: -0.02em;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1.5px solid #e0d9d0;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px 8px 0 0;
        padding: 9px 20px;
        font-size: 14px;
        font-weight: 500;
        color: #7a8fa8;
        border: none;
        letter-spacing: 0.01em;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #2a4060; background: rgba(42,64,96,0.04); }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #1c2f45 !important;
        font-weight: 600 !important;
        border: 1.5px solid #e0d9d0 !important;
        border-bottom: 1.5px solid #ffffff !important;
        margin-bottom: -1.5px;
    }

    .stAlert { border-radius: 14px; font-size: 14px; }

    hr { border: none; border-top: 1.5px solid #e8e2d8; margin: 28px 0; }

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #e4ddd3;
    }

    .stButton > button {
        border-radius: 12px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 15px;
        letter-spacing: 0.01em;
        padding: 10px 28px;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"] {
        background: #1c2f45;
        color: white;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background: #2a4060;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(28,47,69,0.25);
    }

    .warm-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 22px 24px;
        border: 1px solid rgba(28,47,69,0.07);
        box-shadow: 0 2px 10px rgba(28,47,69,0.05);
        margin-bottom: 12px;
        font-family: 'DM Sans', sans-serif;
        font-size: 14.5px;
        color: #1c2f45;
        line-height: 1.6;
    }
    .warm-card-accent   { border-left: 4px solid #c96a28; }
    .warm-card-positive { border-left: 4px solid #2d7d52; }
    .warm-card-warning  { border-left: 4px solid #b84040; }
    .warm-card-neutral  { border-left: 4px solid #4a7fa8; }
    .warm-card .card-label {
        font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; color: #a0afc0; margin-bottom: 8px;
    }
    .warm-card .card-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem; color: #1c2f45; line-height: 1; margin-bottom: 6px;
    }

    .strategy-box {
        background: #ffffff;
        border-radius: 18px;
        padding: 24px 26px;
        height: 100%;
        border: 1px solid rgba(28,47,69,0.07);
        box-shadow: 0 2px 10px rgba(28,47,69,0.05);
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
        color: #3a4f65;
        line-height: 1.65;
    }
    .strategy-box .strategy-label {
        font-size: 11px; font-weight: 600; letter-spacing: 0.1em;
        text-transform: uppercase; margin-bottom: 10px;
    }
    .strategy-box .strategy-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.25rem; color: #1c2f45; margin-bottom: 12px; line-height: 1.25;
    }
    .strategy-a .strategy-label { color: #2a5080; }
    .strategy-a { border-top: 4px solid #2a5080; }
    .strategy-b .strategy-label { color: #c96a28; }
    .strategy-b { border-top: 4px solid #c96a28; }

    .caption-text {
        font-size: 12px; color: #9aacbf; font-style: italic;
        margin-top: 6px; font-family: 'DM Sans', sans-serif;
    }

    .intro-text {
        font-size: 15px; color: #5a6e85; line-height: 1.7;
        margin-bottom: 8px; font-family: 'DM Sans', sans-serif;
    }

    .finding-callout {
        background: #fff9f4;
        border: 1px solid #f0e4d4;
        border-radius: 14px;
        padding: 16px 20px;
        font-size: 14px;
        color: #3a4050;
        line-height: 1.6;
        font-family: 'DM Sans', sans-serif;
    }

    .page-subtitle {
        font-size: 15.5px; color: #6a7e95; margin-top: -6px;
        margin-bottom: 4px; font-style: italic; font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stIconMaterial"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")

@st.cache_resource
def load_models():
    m = {
        "rf_reg_a":  joblib.load("rf_reg_a.pkl"),
        "xgb_clf_a": joblib.load("xgb_clf_a.pkl"),
        "rf_reg_b":  joblib.load("rf_reg_b.pkl"),
        "xgb_clf_b": joblib.load("xgb_clf_b.pkl"),
        "rf_clf_b":  joblib.load("rf_clf_b.pkl"),
    }
    with open("model_b_columns.json", "r") as f:
        m["model_b_columns"] = json.load(f)
    return m

stu    = load_data()
models = load_models()

NAVY  = "#1c2f45"
BLUE  = "#2a5080"
LBLUE = "#4a7fa8"
ORANGE = "#c96a28"
GREEN  = "#2d7d52"
RED    = "#b84040"


def warm_chart(fig, height=None):
    fig.update_layout(
        font_family="DM Sans",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        margin=dict(t=44, b=20, l=10, r=10),
        title_font=dict(size=15, color=NAVY, family="DM Sans"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12, color="#5a6e85")),
    )
    fig.update_xaxes(
        gridcolor="#f0ece6", linecolor="#e0d9d0",
        tickfont=dict(size=11, color="#7a8fa8"),
        title_font=dict(size=12, color="#5a6e85"),
    )
    fig.update_yaxes(
        gridcolor="#f0ece6", linecolor="#e0d9d0",
        tickfont=dict(size=11, color="#7a8fa8"),
        title_font=dict(size=12, color="#5a6e85"),
    )
    if height:
        fig.update_layout(height=height)
    return fig


st.sidebar.markdown(
    "<div style='padding:8px 0 18px; font-family:DM Sans;'>"
    "<div style='font-family:serif; font-size:18px; color:#dce8f5; font-style:italic; margin-bottom:4px;'>Student Performance</div>"
    "<div style='font-size:11px; color:#6a8aaa; letter-spacing:0.08em; text-transform:uppercase;'>Analysis Dashboard</div>"
    "</div>",
    unsafe_allow_html=True
)

page = st.sidebar.radio("", [
    "Home",
    "Dataset Explorer",
    "EDA Visualisations",
    "Model Performance",
    "At-Risk Predictor"
])
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:12px; color:#5a7898; line-height:1.8;'>"
    "Eghonghon Aigbomian<br>20024813<br>Dublin Business School<br>B7IS138"
    "</div>",
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────
if page == "Home":
    st.title("Predicting Student Academic Performance")
    st.markdown(
        "<p class='page-subtitle'>Using Demographic, Behavioural, and Socio-Educational Factors — UCI Student Performance Dataset (Cortez, 2008)</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown(
        "<p class='intro-text'>"
        "This project explores what actually drives student outcomes — not just who scores well, "
        "but <em>why</em>, and whether that can be predicted early enough to matter. "
        "Two machine learning strategies were developed: a full prediction model using all available data including prior grades, "
        "and an early warning model that works without any grade history, "
        "simulating what a school knows at the very start of a term."
        "</p>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", "1,044")
    c2.metric("Variables", "34")
    c3.metric("Maths Students", "395")
    c4.metric("Portuguese Students", "649")

    st.markdown("---")
    st.subheader("Two Modelling Strategies")

    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        st.markdown(
            "<div class='strategy-box strategy-a'>"
            "<div class='strategy-label'>Model A</div>"
            "<div class='strategy-title'>Full Prediction</div>"
            "Uses all available features including prior period grades G1 and G2. "
            "Designed to predict a student's exact final grade as accurately as possible — "
            "the best-case scenario when grade history exists."
            "</div>",
            unsafe_allow_html=True
        )
    with col_b:
        st.markdown(
            "<div class='strategy-box strategy-b'>"
            "<div class='strategy-label'>Model B</div>"
            "<div class='strategy-title'>Early Warning</div>"
            "Removes G1 and G2 to simulate a start-of-term scenario. "
            "Designed to identify students at risk of failing "
            "before any graded assessments have been returned — "
            "the more practically challenging and interesting problem."
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("What the data showed")

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.markdown(
            "<div class='warm-card warm-card-positive'>"
            "<div class='card-label'>Strongest predictor</div>"
            "<div class='card-value'>r = 0.911</div>"
            "G2 alone accounts for <strong>82.5%</strong> of Random Forest feature importance in Model A. "
            "The best predictor of how a student will do is simply how they already did."
            "</div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            "<div class='warm-card warm-card-warning'>"
            "<div class='card-label'>Biggest behavioural signal</div>"
            "<div class='card-value'>&minus;5.25 pts</div>"
            "Students with 3 prior failures average <strong>6.80</strong>, "
            "compared to <strong>12.05</strong> for those with none. "
            "A single failure already drops the average below the pass mark."
            "</div>",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            "<div class='warm-card warm-card-neutral'>"
            "<div class='card-label'>Model results</div>"
            "<div class='card-value'>R&sup2; 0.827</div>"
            "Best regression: <strong>Random Forest Model A</strong>.<br>"
            "Best early warning: <strong>RF Model B, R&sup2; = 0.239</strong> — "
            "significantly harder without grade history, but not impossible."
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Dataset at a glance")

    col_left, col_right = st.columns(2, gap="medium")
    with col_left:
        subj_counts = stu["subject"].value_counts().reset_index()
        subj_counts.columns = ["Subject", "Count"]
        fig_subj = px.pie(
            subj_counts, names="Subject", values="Count",
            title="Students by Subject",
            color_discrete_sequence=[BLUE, ORANGE], hole=0.45
        )
        fig_subj.update_traces(textfont=dict(family="DM Sans", size=13))
        warm_chart(fig_subj)
        st.plotly_chart(fig_subj, use_container_width=True)

    with col_right:
        higher_counts = stu["higher"].value_counts().reset_index()
        higher_counts.columns = ["Plans for Higher Education", "Count"]
        fig_higher = px.pie(
            higher_counts, names="Plans for Higher Education", values="Count",
            title="Higher Education Aspiration",
            color_discrete_sequence=[GREEN, RED], hole=0.45
        )
        fig_higher.update_traces(textfont=dict(family="DM Sans", size=13))
        warm_chart(fig_higher)
        st.plotly_chart(fig_higher, use_container_width=True)


# ─────────────────────────────────────────────
# DATASET EXPLORER
# ─────────────────────────────────────────────
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    st.markdown(
        "<p class='intro-text'>Browse the full student dataset. Filter by subject, sex, or at-risk status and everything updates live.</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        subject_filter = st.selectbox("Subject", ["All", "math", "portuguese"])
    with col2:
        sex_filter = st.selectbox("Sex", ["All", "F", "M"])
    with col3:
        risk_filter = st.selectbox("At-Risk Status (G3 < 10)", ["All", "At Risk", "Not At Risk"])

    filtered = stu.copy()
    if subject_filter != "All":
        filtered = filtered[filtered["subject"] == subject_filter]
    if sex_filter != "All":
        filtered = filtered[filtered["sex"] == sex_filter]
    if risk_filter == "At Risk":
        filtered = filtered[filtered["G3"] < 10]
    elif risk_filter == "Not At Risk":
        filtered = filtered[filtered["G3"] >= 10]

    st.markdown(
        f"<p class='intro-text'>Showing <strong>{len(filtered)}</strong> of 1,044 students</p>",
        unsafe_allow_html=True
    )

    display_cols = [
        "school", "sex", "age", "address", "Medu", "Fedu", "Mjob", "Fjob",
        "studytime", "failures", "absences", "higher", "Walc", "Dalc",
        "G1", "G2", "G3", "subject"
    ]
    st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.subheader("Summary Statistics")
    st.dataframe(
        filtered[["age", "studytime", "failures", "absences", "G1", "G2", "G3"]]
        .describe().round(2),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Grade Distribution")

    col_hist, col_box = st.columns(2, gap="medium")
    with col_hist:
        fig_hist = px.histogram(
            filtered, x="G3", nbins=20,
            title="Distribution of Final Grades (G3)",
            labels={"G3": "Final Grade (G3)"},
            color_discrete_sequence=[BLUE]
        )
        fig_hist.update_layout(bargap=0.12, showlegend=False)
        warm_chart(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
        fig_box_subj = px.box(
            filtered, x="subject", y="G3", color="subject",
            title="G3 by Subject",
            labels={"G3": "Final Grade (G3)", "subject": "Subject"},
            color_discrete_map={"math": BLUE, "portuguese": ORANGE}
        )
        fig_box_subj.update_layout(showlegend=False)
        warm_chart(fig_box_subj)
        st.plotly_chart(fig_box_subj, use_container_width=True)

    st.markdown("---")
    st.subheader("Pass / Fail Breakdown")

    filtered_viz = filtered.copy()
    filtered_viz["Status"] = filtered_viz["G3"].apply(
        lambda x: "At Risk (G3 < 10)" if x < 10 else "Passing (G3 >= 10)"
    )
    status_counts = filtered_viz["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    fig_status = px.bar(
        status_counts, x="Status", y="Count",
        color="Status",
        color_discrete_map={"At Risk (G3 < 10)": RED, "Passing (G3 >= 10)": GREEN},
        title="Students At Risk vs Passing",
        text="Count"
    )
    fig_status.update_traces(textposition="outside", marker_line_width=0)
    fig_status.update_layout(showlegend=False)
    warm_chart(fig_status)
    st.plotly_chart(fig_status, use_container_width=True)


# ─────────────────────────────────────────────
# EDA VISUALISATIONS
# ─────────────────────────────────────────────
elif page == "EDA Visualisations":
    st.title("Exploratory Data Analysis")
    st.markdown(
        "<p class='intro-text'>A full exploration of the dataset — from single-variable distributions to cross-variable relationships and lifestyle patterns.</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "  Variable vs G3  ",
        "  Correlation Heatmap  ",
        "  Grade Trajectories  ",
        "  Lifestyle & Behaviour  "
    ])

    with tab1:
        st.subheader("Explore Any Variable Against Final Grade")
        st.markdown(
            "<p class='intro-text'>Pick any variable from the list. The boxplot shows how final grades are distributed within each group, and the table shows the group averages.</p>",
            unsafe_allow_html=True
        )

        variable = st.selectbox("Variable to explore:", [
            "failures", "studytime", "Walc", "Dalc", "goout", "freetime",
            "Medu", "Fedu", "higher", "subject", "sex", "address",
            "Mjob", "Fjob", "health", "romantic", "internet", "activities",
            "schoolsup", "paid"
        ])

        group_means = (
            stu.groupby(variable)["G3"]
            .mean().reset_index()
            .rename(columns={"G3": "Mean G3"})
        )
        group_means["Mean G3"] = group_means["Mean G3"].round(2)

        col_plot, col_table = st.columns([2, 1], gap="medium")
        with col_plot:
            fig_box = px.box(
                stu, x=variable, y="G3",
                title=f"G3 vs {variable}",
                labels={"G3": "Final Grade (G3)", variable: variable},
                color=variable,
                color_discrete_sequence=[BLUE, ORANGE, GREEN, LBLUE, "#7a5080", "#508070"]
            )
            fig_box.update_layout(showlegend=False)
            warm_chart(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)

        with col_table:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Group averages**")
            st.dataframe(group_means, use_container_width=True, height=280)

        interpretations = {
            "failures":   "Clear negative trend. Mean G3 drops from 12.05 with no failures to 6.80 with three. Correlation: -0.383. This is the strongest non-grade predictor in the dataset — even one prior failure drops the average below the pass mark.",
            "studytime":  "Weak positive trend. Mean G3 rises from 10.58 at category 1 to 12.49 at category 3, then dips slightly to 12.27 at category 4. More study generally helps, but it is a modest effect on its own.",
            "Walc":       "Weak negative relationship. Higher weekend alcohol consumption is associated with marginally lower grades. Walc and Dalc are strongly correlated with each other, capturing the same underlying behaviour.",
            "Dalc":       "Similar to Walc. Students who drink on weekdays also tend to drink heavily at weekends. Neither variable is a dominant grade predictor in isolation.",
            "goout":      "Very weak negative trend. Going out frequency has minimal direct impact on final grades.",
            "freetime":   "No consistent pattern. Mean grades fluctuate between 10.92 and 12.28 without a clear direction — what students do with free time seems to matter more than how much they have.",
            "Medu":       "Moderate positive relationship. Students with more educated mothers tend to score higher. Correlation with G3: 0.201.",
            "Fedu":       "Similar to mother's education but slightly weaker. Correlation with G3: 0.160. Both parental education variables appear in the top Model B features.",
            "higher":     "Strong difference. Students planning to pursue higher education average 11.62 compared to 8.35 for those who do not — a gap of over 3 grade points.",
            "subject":    "Moderate difference. Portuguese students average 11.91 compared to 10.42 for Mathematics. Subject appears in feature importance charts for both model sets.",
            "sex":        "Very weak. Females average 11.45, males 11.20 — a gap of just 0.25 grade points. Sex is not a meaningful predictor here.",
            "address":    "Weak. Urban students show a marginally higher median grade than rural students, possibly reflecting easier access to resources.",
            "Mjob":       "Moderate variation. Children of mothers in health and teaching roles tend to perform best. Mothers recorded as at-home are associated with the lowest averages.",
            "Fjob":       "Similar pattern to mother's job. Father working as a teacher is associated with the highest average G3.",
            "health":     "Very weak. Self-reported health status shows minimal relationship with final grade.",
            "romantic":   "Small negative difference. Students in romantic relationships score marginally lower on average.",
            "internet":   "Weak positive. Students with home internet access tend to perform slightly better.",
            "activities": "Negligible. Extracurricular activities show almost no impact on final grade.",
            "schoolsup":  "Counterintuitive. Students receiving extra school support actually score lower on average — because support goes to students already struggling, not because support causes lower grades.",
            "paid":       "Negligible. Paid tutoring shows very little difference in grade outcome."
        }

        if variable in interpretations:
            st.markdown(
                f"<div class='finding-callout'><strong>Finding:</strong> {interpretations[variable]}</div>",
                unsafe_allow_html=True
            )

    with tab2:
        st.subheader("Correlation Heatmap")
        st.markdown(
            "<p class='intro-text'>Values close to 1 mean a strong positive relationship. Values close to -1 mean a strong negative one. Values near 0 mean little to no linear relationship.</p>",
            unsafe_allow_html=True
        )

        num_cols = stu.select_dtypes(include=["int64", "float64"]).columns.tolist()
        corr = stu[num_cols].corr().round(2)

        fig_heat = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Correlation Matrix — All Numeric Variables",
            text_auto=True,
            aspect="auto"
        )
        fig_heat.update_traces(textfont=dict(size=10, family="DM Sans"))
        warm_chart(fig_heat, height=680)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")
        st.subheader("Correlations with G3 — Ranked")

        g3_corr = (
            stu[num_cols].corr()["G3"].drop("G3")
            .sort_values(ascending=False).reset_index()
        )
        g3_corr.columns = ["Variable", "Correlation with G3"]
        g3_corr["Correlation with G3"] = g3_corr["Correlation with G3"].round(3)

        fig_ranked = px.bar(
            g3_corr, x="Variable", y="Correlation with G3",
            color="Correlation with G3",
            color_continuous_scale="RdBu_r",
            title="Each Variable's Correlation with G3",
            range_color=[-1, 1], text="Correlation with G3"
        )
        fig_ranked.add_hline(y=0, line_dash="dot", line_color="#c0b8b0", line_width=1.5)
        fig_ranked.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
        fig_ranked.update_layout(coloraxis_showscale=False, xaxis_tickangle=-40)
        warm_chart(fig_ranked)
        st.plotly_chart(fig_ranked, use_container_width=True)

        st.markdown(
            "<div class='finding-callout'>"
            "G2 (0.911) and G1 (0.809) dominate by a wide margin. Prior failures (-0.383) is the strongest negative predictor. "
            "Most behavioural variables sit between -0.13 and 0.20 — weak on their own, "
            "but tree-based models can detect non-linear combinations that correlation alone misses."
            "</div>",
            unsafe_allow_html=True
        )

    with tab3:
        st.subheader("Grade Trajectories")
        st.markdown(
            "<p class='intro-text'>How student grades evolve across the three grading periods, and how strongly each prior period predicts the final outcome.</p>",
            unsafe_allow_html=True
        )

        col_sel, col_chk = st.columns([2, 1])
        with col_sel:
            grade_var = st.selectbox("Compare G3 against:", ["G1", "G2"])
        with col_chk:
            subject_colour = st.checkbox("Colour by subject", value=True)

        fig_scatter = px.scatter(
            stu, x=grade_var, y="G3",
            color="subject" if subject_colour else None,
            opacity=0.45, trendline="ols",
            title=f"G3 vs {grade_var} with Regression Line",
            labels={grade_var: f"{grade_var} (Period Grade)", "G3": "Final Grade (G3)"},
            color_discrete_map={"math": BLUE, "portuguese": ORANGE}
        )
        warm_chart(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

        corr_val = stu[grade_var].corr(stu["G3"])
        st.markdown(
            f"<div class='finding-callout'>"
            f"Correlation between {grade_var} and G3: <strong>{corr_val:.3f}</strong> — "
            f"{'an extremely strong positive relationship.' if corr_val > 0.85 else 'a very strong positive relationship.'} "
            f"Students are remarkably consistent across grading periods."
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("Average Grade Across All Three Periods by Subject")

        avg_grades = stu.groupby("subject")[["G1", "G2", "G3"]].mean().reset_index()
        avg_melted = avg_grades.melt(id_vars="subject", var_name="Period", value_name="Average Grade")

        fig_trend = px.line(
            avg_melted, x="Period", y="Average Grade", color="subject",
            markers=True, title="Average Grade per Period by Subject",
            color_discrete_map={"math": BLUE, "portuguese": ORANGE},
            labels={"Average Grade": "Average Grade (0-20)"}
        )
        fig_trend.update_traces(line_width=3, marker_size=9)
        warm_chart(fig_trend)
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("---")
        st.subheader("Study Time by Subject")

        fig_study_subj = px.box(
            stu, x="studytime", y="G3", color="subject",
            title="G3 vs Study Time by Subject",
            labels={"studytime": "Weekly Study Time", "G3": "Final Grade (G3)"},
            color_discrete_map={"math": BLUE, "portuguese": ORANGE}
        )
        warm_chart(fig_study_subj)
        st.plotly_chart(fig_study_subj, use_container_width=True)

        st.markdown(
            "<div class='finding-callout'>"
            "Portuguese students consistently score higher than Maths students at every study time level. "
            "A Maths student studying more than 10 hours per week achieves similar grades to a Portuguese student studying 2 to 5 hours — "
            "subject difficulty plays an independent role beyond raw effort."
            "</div>",
            unsafe_allow_html=True
        )

    with tab4:
        st.subheader("Lifestyle and Behavioural Patterns")
        st.markdown(
            "<p class='intro-text'>How alcohol consumption, social behaviour, and aspiration relate to absences and grades.</p>",
            unsafe_allow_html=True
        )

        col_left, col_right = st.columns(2, gap="medium")
        with col_left:
            fig_walc = px.box(
                stu, x="Walc", y="absences",
                title="Absences vs Weekend Alcohol",
                labels={"Walc": "Weekend Alcohol (1=very low, 5=very high)", "absences": "Absences"},
                color="Walc",
                color_discrete_sequence=[BLUE, LBLUE, ORANGE, "#c96a28", RED]
            )
            fig_walc.update_layout(showlegend=False)
            warm_chart(fig_walc)
            st.plotly_chart(fig_walc, use_container_width=True)

        with col_right:
            fig_dalc = px.box(
                stu, x="Dalc", y="absences",
                title="Absences vs Weekday Alcohol",
                labels={"Dalc": "Weekday Alcohol (1=very low, 5=very high)", "absences": "Absences"},
                color="Dalc",
                color_discrete_sequence=[BLUE, LBLUE, ORANGE, "#c96a28", RED]
            )
            fig_dalc.update_layout(showlegend=False)
            warm_chart(fig_dalc)
            st.plotly_chart(fig_dalc, use_container_width=True)

        st.markdown(
            "<div class='finding-callout'>"
            "Higher alcohol consumption is associated with slightly more absences, "
            "but neither variable independently drives poor grades in a strong or consistent way. "
            "Together they likely reflect a broader pattern of disengagement rather than directly causing lower outcomes."
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")

        col_l2, col_r2 = st.columns(2, gap="medium")
        with col_l2:
            fig_goout_walc = px.scatter(
                stu, x="goout", y="Walc", color="G3",
                color_continuous_scale="RdYlGn",
                title="Going Out vs Weekend Alcohol — coloured by G3",
                labels={"goout": "Going Out Frequency", "Walc": "Weekend Alcohol"},
                opacity=0.65
            )
            warm_chart(fig_goout_walc)
            st.plotly_chart(fig_goout_walc, use_container_width=True)

        with col_r2:
            aspiration_data = stu.groupby(["higher", "failures"])["G3"].mean().reset_index()
            aspiration_data.columns = ["Higher Education", "Prior Failures", "Mean G3"]
            aspiration_data["Mean G3"] = aspiration_data["Mean G3"].round(2)

            fig_aspiration = px.bar(
                aspiration_data, x="Prior Failures", y="Mean G3",
                color="Higher Education", barmode="group",
                title="Does Aspiration Soften the Effect of Prior Failures?",
                color_discrete_map={"yes": GREEN, "no": RED},
                labels={"Mean G3": "Mean Final Grade (G3)"},
                text="Mean G3"
            )
            fig_aspiration.update_traces(texttemplate="%{text:.1f}", textposition="outside", marker_line_width=0)
            warm_chart(fig_aspiration)
            st.plotly_chart(fig_aspiration, use_container_width=True)

        st.markdown(
            "<div class='finding-callout'>"
            "Students who plan to pursue higher education consistently outperform those who do not at every failure level. "
            "Even students with two or three prior failures but strong educational aspirations score higher on average "
            "than non-aspiring students with zero failures. "
            "Motivational support may be as important as academic remediation."
            "</div>",
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────
# MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "Model Performance":
    st.title("Model Performance Comparison")
    st.markdown(
        "<p class='intro-text'>"
        "Five regression algorithms and three classification algorithms were applied to both model sets, "
        "giving ten regression and six classification models in total."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "  Regression Models  ",
        "  Classification Models  ",
        "  Feature Importance  "
    ])

    with tab1:
        st.subheader("Regression Results")

        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            st.markdown("**Model A — With Prior Grades (G1, G2)**")
            reg_a = pd.DataFrame({
                "Model":  ["Linear A", "Ridge A", "Lasso A", "RF A", "XGB A"],
                "R²":     [0.800, 0.800, 0.807, 0.827, 0.823],
                "RMSE":   [1.760, 1.758, 1.726, 1.636, 1.654]
            })
            st.dataframe(reg_a.set_index("Model"), use_container_width=True)
            st.markdown(
                "<div class='caption-text'>Best: Random Forest A — R² = 0.827, RMSE = 1.636</div>",
                unsafe_allow_html=True
            )

        with col_b:
            st.markdown("**Model B — Without Prior Grades (Early Warning)**")
            reg_b = pd.DataFrame({
                "Model":  ["Linear B", "Ridge B", "Lasso B", "RF B", "XGB B"],
                "R²":     [0.119, 0.121, 0.136, 0.239, 0.106],
                "RMSE":   [3.692, 3.686, 3.654, 3.431, 3.719]
            })
            st.dataframe(reg_b.set_index("Model"), use_container_width=True)
            st.markdown(
                "<div class='caption-text'>Best: RF B — R² = 0.239 &nbsp;|&nbsp; Worst: XGB B — R² = 0.106</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.subheader("R² Score Comparison — All Models")

        all_reg = pd.DataFrame({
            "Model": [
                "Linear A", "Ridge A", "Lasso A", "RF A", "XGB A",
                "Linear B", "Ridge B", "Lasso B", "RF B", "XGB B"
            ],
            "R²": [0.800, 0.800, 0.807, 0.827, 0.823,
                   0.119, 0.121, 0.136, 0.239, 0.106],
            "Set": ["Model A"] * 5 + ["Model B"] * 5
        })

        fig_r2 = px.bar(
            all_reg, x="Model", y="R²", color="Set",
            barmode="group",
            title="R² Scores — Model A vs Model B",
            color_discrete_map={"Model A": BLUE, "Model B": ORANGE},
            text="R²"
        )
        fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
        fig_r2.update_layout(yaxis_range=[0, 1.08], xaxis_tickangle=-28, yaxis_title="R² Score")
        warm_chart(fig_r2)
        st.plotly_chart(fig_r2, use_container_width=True)

        st.markdown(
            "<div class='finding-callout'>"
            "The drop from Model A (0.80-0.83) to Model B (0.11-0.24) makes one thing clear: prior grades dominate prediction. "
            "Without them, exact grade forecasting is significantly harder. "
            "Random Forest remains the strongest performer in both sets. "
            "XGBoost performs worst in Model B — likely overfitting without the grade features to anchor it."
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("Actual vs Predicted — Random Forest Regression")
        st.markdown(
            "<p class='intro-text'>If the model were perfect, every dot would sit on the red dashed line. The tighter the cluster, the better.</p>",
            unsafe_allow_html=True
        )

        model_choice = st.selectbox("Select model:", ["Model A (RF)", "Model B (RF)"])

        from sklearn.model_selection import train_test_split

        stu_model = stu.copy()
        stu_model["at_risk"] = (stu_model["G3"] < 10).astype(int)
        y_reg = stu_model["G3"]

        X_full      = pd.get_dummies(stu_model.drop(["G3", "at_risk"], axis=1), drop_first=True)
        X_no_grades = pd.get_dummies(stu_model.drop(["G3", "G1", "G2", "at_risk"], axis=1), drop_first=True)

        if model_choice == "Model A (RF)":
            Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(X_full, y_reg, test_size=0.2, random_state=42)
            y_pred      = models["rf_reg_a"].predict(Xf_te)
            y_true      = yf_te
            chart_title = "RF Regression Model A — Actual vs Predicted"
        else:
            Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(X_no_grades, y_reg, test_size=0.2, random_state=42)
            y_pred      = models["rf_reg_b"].predict(Xn_te)
            y_true      = yn_te
            chart_title = "RF Regression Model B — Actual vs Predicted"

        avp_df = pd.DataFrame({"Actual G3": y_true.values, "Predicted G3": y_pred})
        fig_avp = px.scatter(
            avp_df, x="Actual G3", y="Predicted G3",
            opacity=0.5, title=chart_title,
            color_discrete_sequence=[BLUE]
        )
        fig_avp.add_shape(
            type="line", x0=0, y0=0, x1=20, y1=20,
            line=dict(color=RED, dash="dot", width=2)
        )
        fig_avp.add_annotation(
            x=17.5, y=15.5, text="Perfect prediction",
            showarrow=False, font=dict(color=RED, size=11, family="DM Sans")
        )
        warm_chart(fig_avp)
        st.plotly_chart(fig_avp, use_container_width=True)

        if model_choice == "Model A (RF)":
            st.success(
                "Model A points cluster tightly around the diagonal. "
                "The main exception is students who scored 0 — the model over-predicts these, "
                "likely because they represent dropouts rather than genuine low performers."
            )
        else:
            st.warning(
                "Model B points are considerably more scattered. The model tends to predict "
                "middle-range grades regardless of actual score, confirming that without G1 and G2 "
                "it cannot reliably differentiate strong from weak performers."
            )

    with tab2:
        st.subheader("Classification Results")
        st.markdown(
            "<p class='intro-text'>Students scoring below 10 were labelled at-risk. The threshold of 10 reflects the Portuguese minimum pass mark.</p>",
            unsafe_allow_html=True
        )

        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            st.markdown("**Model A — With Prior Grades**")
            clf_a = pd.DataFrame({
                "Model":     ["Logistic Reg A", "RF A", "XGB A"],
                "Accuracy":  [0.885, 0.885, 0.895],
                "Precision": [0.860, 0.891, 0.880],
                "Recall":    [0.717, 0.683, 0.733],
                "F1":        [0.782, 0.774, 0.800]
            })
            st.dataframe(clf_a.set_index("Model"), use_container_width=True)
            st.markdown(
                "<div class='caption-text'>Best: XGBoost A — Accuracy 89.5%, F1 0.800</div>",
                unsafe_allow_html=True
            )

        with col_b:
            st.markdown("**Model B — Without Prior Grades**")
            clf_b = pd.DataFrame({
                "Model":     ["Logistic Reg B", "RF B", "XGB B"],
                "Accuracy":  [0.737, 0.742, 0.732],
                "Precision": [0.609, 0.625, 0.553],
                "Recall":    [0.233, 0.250, 0.350],
                "F1":        [0.337, 0.357, 0.429]
            })
            st.dataframe(clf_b.set_index("Model"), use_container_width=True)
            st.markdown(
                "<div class='caption-text'>Best F1: XGBoost B — 0.429 &nbsp;|&nbsp; Best recall: XGBoost B — 0.350</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        metric_choice = st.selectbox(
            "Select a metric to compare across all six models:",
            ["Accuracy", "Precision", "Recall", "F1"]
        )

        metric_data = pd.DataFrame({
            "Model":     ["LogReg A", "RF A", "XGB A", "LogReg B", "RF B", "XGB B"],
            "Accuracy":  [0.885, 0.885, 0.895, 0.737, 0.742, 0.732],
            "Precision": [0.860, 0.891, 0.880, 0.609, 0.625, 0.553],
            "Recall":    [0.717, 0.683, 0.733, 0.233, 0.250, 0.350],
            "F1":        [0.782, 0.774, 0.800, 0.337, 0.357, 0.429],
            "Set":       ["Model A"] * 3 + ["Model B"] * 3
        })

        fig_clf = px.bar(
            metric_data, x="Model", y=metric_choice, color="Set",
            title=f"{metric_choice} — All Classification Models",
            color_discrete_map={"Model A": BLUE, "Model B": ORANGE},
            text=metric_choice
        )
        fig_clf.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
        fig_clf.update_layout(yaxis_range=[0, 1.18])
        warm_chart(fig_clf)
        st.plotly_chart(fig_clf, use_container_width=True)

        if metric_choice == "Recall":
            st.markdown(
                "<div class='finding-callout'>"
                "Recall is the most important metric in this context — it measures how many genuinely at-risk students the model catches. "
                "A low recall means students who need support are being missed. "
                "Model B's recall of 0.350 means 65% of at-risk students go undetected without prior grade data."
                "</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.subheader("Confusion Matrix")
        st.markdown(
            "<p class='intro-text'>"
            "False negatives — at-risk students the model missed — are the costly error. "
            "A missed student receives no support. False positives are less harmful: at worst, a student gets an unnecessary check-in."
            "</p>",
            unsafe_allow_html=True
        )

        cm_choice = st.selectbox("Select model:", ["Model A (XGBoost)", "Model B (XGBoost)"])

        stu_cm = stu.copy()
        stu_cm["at_risk"] = (stu_cm["G3"] < 10).astype(int)
        y_clf_full = stu_cm["at_risk"]

        X_full_cm      = pd.get_dummies(stu_cm.drop(["G3", "at_risk"], axis=1), drop_first=True)
        X_no_grades_cm = pd.get_dummies(stu_cm.drop(["G3", "G1", "G2", "at_risk"], axis=1), drop_first=True)

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix

        if cm_choice == "Model A (XGBoost)":
            Xf_c_tr, Xf_c_te, yf_c_tr, yf_c_te = train_test_split(
                X_full_cm, y_clf_full, test_size=0.2, random_state=42
            )
            y_pred_cm = models["xgb_clf_a"].predict(Xf_c_te)
            y_true_cm = yf_c_te
        else:
            Xn_c_tr, Xn_c_te, yn_c_tr, yn_c_te = train_test_split(
                X_no_grades_cm, y_clf_full, test_size=0.2, random_state=42
            )
            y_pred_cm = models["xgb_clf_b"].predict(Xn_c_te)
            y_true_cm = yn_c_te

        cm_arr = confusion_matrix(y_true_cm, y_pred_cm)
        labels = ["Not At Risk", "At Risk"]

        fig_cm = px.imshow(
            cm_arr, x=labels, y=labels,
            text_auto=True,
            color_continuous_scale=[[0, "#f0f4f8"], [1, BLUE]],
            title=f"Confusion Matrix — XGBoost {cm_choice}",
            labels=dict(x="Predicted", y="Actual")
        )
        fig_cm.update_traces(textfont=dict(size=18, family="DM Serif Display"))
        warm_chart(fig_cm, height=380)
        fig_cm.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

        tn, fp, fn, tp = cm_arr.ravel()
        col_tn, col_tp, col_fp, col_fn = st.columns(4)
        col_tn.metric("True Negatives",  tn, help="Safe students correctly cleared")
        col_tp.metric("True Positives",  tp, help="At-risk students correctly flagged")
        col_fp.metric("False Positives", fp, help="Safe students wrongly flagged")
        col_fn.metric("False Negatives", fn, help="At-risk students missed")

        st.markdown("<br>", unsafe_allow_html=True)
        if cm_choice == "Model A (XGBoost)":
            st.success(
                f"Model A correctly identified {tp} of {tp + fn} at-risk students (recall 73.3%). "
                f"Only {fn} were missed — an acceptable rate for a practical intervention tool."
            )
        else:
            st.warning(
                f"Model B missed {fn} of {tp + fn} at-risk students (recall 35.0%). "
                f"The {tp} correctly flagged still represent a meaningful early signal, "
                f"but further development is needed before real-world use."
            )

    with tab3:
        st.subheader("Feature Importance")
        st.markdown(
            "<p class='intro-text'>"
            "Feature importance shows how much each variable contributed to the model's decisions. "
            "The contrast between Model A and Model B here is one of the most interesting findings of the project."
            "</p>",
            unsafe_allow_html=True
        )

        model_fi = st.selectbox("Select model:", [
            "RF Regression — Model A",
            "RF Regression — Model B",
        ])

        if model_fi == "RF Regression — Model A":
            fi_data = pd.DataFrame({
                "Feature": [
                    "G2", "absences", "G1", "studytime", "age", "goout",
                    "subject_portuguese", "health", "reason_home", "freetime",
                    "sex_M", "Fjob_services", "famrel", "Walc", "Medu"
                ],
                "Importance": [
                    0.825, 0.059, 0.010, 0.010, 0.008, 0.007,
                    0.007, 0.006, 0.004, 0.004, 0.004, 0.004, 0.004, 0.003, 0.003
                ]
            }).sort_values("Importance", ascending=True)
            colour = BLUE
        else:
            fi_data = pd.DataFrame({
                "Feature": [
                    "failures", "absences", "subject_portuguese", "goout",
                    "schoolsup_yes", "age", "health", "freetime",
                    "studytime", "Medu", "Walc", "Fedu",
                    "famrel", "traveltime", "higher_yes"
                ],
                "Importance": [
                    0.187, 0.140, 0.043, 0.040, 0.038, 0.034, 0.034, 0.032,
                    0.032, 0.032, 0.029, 0.027, 0.025, 0.025, 0.023
                ]
            }).sort_values("Importance", ascending=True)
            colour = ORANGE

        fig_fi = px.bar(
            fi_data, x="Importance", y="Feature",
            orientation="h",
            title=f"Top 15 Features — {model_fi}",
            color_discrete_sequence=[colour],
            text="Importance"
        )
        fig_fi.update_traces(
            texttemplate="%{text:.3f}", textposition="outside",
            marker_line_width=0, marker_color=colour
        )
        fig_fi.update_layout(yaxis=dict(tickfont=dict(size=12, family="DM Sans")))
        warm_chart(fig_fi, height=520)
        st.plotly_chart(fig_fi, use_container_width=True)

        if model_fi == "RF Regression — Model A":
            st.markdown(
                "<div class='finding-callout'>"
                "G2 accounts for 82.5% of total importance. The model is essentially learning that "
                "the second period grade predicts the final grade, with small adjustments from everything else. "
                "This explains why all Model A algorithms perform within a narrow band — "
                "the problem reduces to a near-linear function of one dominant variable."
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='finding-callout'>"
                "Without grade history, the picture changes completely. "
                "Prior failures (18.7%) and absences (14.0%) become the leading signals. "
                "Absences carry almost no linear correlation with G3 (-0.046), "
                "yet the Random Forest detects a meaningful non-linear pattern — students with very high absences "
                "are disproportionately likely to fail. "
                "Failures, aspiration, and school support are all recorded at enrolment, making them genuinely actionable."
                "</div>",
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────
# AT-RISK PREDICTOR
# ─────────────────────────────────────────────
elif page == "At-Risk Predictor":
    st.title("Early Warning: At-Risk Student Predictor")
    st.markdown(
        "<p class='intro-text'>"
        "Fill in the details below and the model will predict whether this student is likely to score below 10 — "
        "using only demographic and behavioural information, with no prior grade history required. "
        "This uses the trained Model B XGBoost classifier."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("**Academic History**")
        failures  = st.slider("Prior Failures", 0, 3, 0)
        studytime = st.select_slider(
            "Weekly Study Time", options=[1, 2, 3, 4], value=2,
            format_func=lambda x: {
                1: "Less than 2 hours", 2: "2 to 5 hours",
                3: "5 to 10 hours",     4: "More than 10 hours"
            }[x]
        )
        absences  = st.slider("Number of Absences", 0, 75, 4)
        schoolsup = st.selectbox("Receives Extra School Support", ["no", "yes"])
        higher    = st.selectbox("Plans for Higher Education", ["yes", "no"])

    with col2:
        st.markdown("**Family Background**")
        Medu = st.select_slider(
            "Mother's Education Level", options=[0, 1, 2, 3, 4], value=2,
            format_func=lambda x: {
                0: "None", 1: "Primary school", 2: "Middle school",
                3: "Secondary school", 4: "Higher education"
            }[x]
        )
        Fedu = st.select_slider(
            "Father's Education Level", options=[0, 1, 2, 3, 4], value=2,
            format_func=lambda x: {
                0: "None", 1: "Primary school", 2: "Middle school",
                3: "Secondary school", 4: "Higher education"
            }[x]
        )
        Mjob   = st.selectbox("Mother's Occupation", ["at_home", "health", "other", "services", "teacher"])
        Fjob   = st.selectbox("Father's Occupation", ["at_home", "health", "other", "services", "teacher"])
        famrel = st.slider("Family Relationship Quality (1 = very bad, 5 = excellent)", 1, 5, 4)

    with col3:
        st.markdown("**Lifestyle**")
        goout    = st.slider("Going Out with Friends (1 = very low, 5 = very high)", 1, 5, 2)
        Walc     = st.slider("Weekend Alcohol (1 = very low, 5 = very high)", 1, 5, 1)
        Dalc     = st.slider("Weekday Alcohol (1 = very low, 5 = very high)", 1, 5, 1)
        freetime = st.slider("Free Time After School (1 = very low, 5 = very high)", 1, 5, 3)
        health   = st.slider("Health Status (1 = very bad, 5 = very good)", 1, 5, 3)
        subject  = st.selectbox("Subject", ["math", "portuguese"])
        sex      = st.selectbox("Sex", ["F", "M"])
        address  = st.selectbox("Address Type", ["U (Urban)", "R (Rural)"])
        age      = st.slider("Age", 15, 22, 17)
        school   = st.selectbox("School", ["GP", "MS"])

    st.markdown("---")

    if st.button("Run Prediction", type="primary"):
        address_val = "U" if "Urban" in address else "R"

        input_dict = {
            "school": school, "sex": sex, "age": age,
            "address": address_val, "famsize": "GT3", "Pstatus": "T",
            "Medu": Medu, "Fedu": Fedu, "Mjob": Mjob, "Fjob": Fjob,
            "reason": "course", "guardian": "mother", "traveltime": 1,
            "studytime": studytime, "failures": failures,
            "schoolsup": schoolsup, "famsup": "yes", "paid": "no",
            "activities": "no", "nursery": "yes", "higher": higher,
            "internet": "yes", "romantic": "no", "famrel": famrel,
            "freetime": freetime, "goout": goout, "Dalc": Dalc,
            "Walc": Walc, "health": health, "absences": absences,
            "subject": subject
        }

        input_df      = pd.DataFrame([input_dict])
        input_dummies = pd.get_dummies(input_df, drop_first=True)

        model_b_columns = models["model_b_columns"]
        for col in model_b_columns:
            if col not in input_dummies.columns:
                input_dummies[col] = 0
        input_dummies = input_dummies[model_b_columns]

        prediction  = models["xgb_clf_b"].predict(input_dummies)[0]
        probability = models["xgb_clf_b"].predict_proba(input_dummies)[0]

        col_result, col_prob = st.columns([1, 1], gap="medium")

        with col_result:
            st.markdown("### Result")
            if prediction == 1:
                st.error(
                    "At Risk\n\n"
                    "Based on the information provided, this student is predicted to score below 10."
                )
            else:
                st.success(
                    "Not At Risk\n\n"
                    "Based on the information provided, this student is predicted to pass."
                )

            risk_pct = round(probability[1] * 100, 1)
            pass_pct = round(probability[0] * 100, 1)
            st.markdown(f"**Probability of failing:** {risk_pct}%")
            st.markdown(f"**Probability of passing:** {pass_pct}%")

            st.markdown("<br>**Factors influencing this prediction:**", unsafe_allow_html=True)
            if failures > 0:
                st.markdown(f"- {failures} prior failure(s) — the strongest early warning signal in Model B")
            if absences > 10:
                st.markdown(f"- {absences} absences — above the dataset average")
            if higher == "no":
                st.markdown("- No plans for higher education — associated with lower average grades")
            if Walc >= 4 or Dalc >= 3:
                st.markdown("- Elevated alcohol consumption")
            if failures == 0 and absences <= 5 and higher == "yes":
                st.markdown("- No major risk factors detected from the information provided")

        with col_prob:
            st.markdown("### Confidence")
            prob_df = pd.DataFrame({
                "Outcome":     ["Passing", "At Risk"],
                "Probability": [round(probability[0], 4), round(probability[1], 4)]
            })
            fig_prob = px.bar(
                prob_df, x="Outcome", y="Probability",
                color="Outcome",
                color_discrete_map={"Passing": GREEN, "At Risk": RED},
                title="Model Confidence",
                text="Probability"
            )
            fig_prob.update_traces(
                texttemplate="%{text:.1%}", textposition="outside", marker_line_width=0
            )
            fig_prob.update_layout(
                yaxis_range=[0, 1.3], showlegend=False, yaxis_tickformat=".0%"
            )
            warm_chart(fig_prob)
            st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("---")
        st.markdown(
            "<div class='finding-callout'>"
            "This uses the Model B XGBoost classifier, trained on demographic and behavioural data only. "
            "Model B recall = 0.350 — around 65% of genuinely at-risk students are not detected. "
            "This is a decision-support tool, not a definitive assessment of any student's outcome."
            "</div>",
            unsafe_allow_html=True
        )
