"""
WorkBalance AI — Streamlit app
AI-based discovery of work-behavior profiles across 85 cultures

Run with:  streamlit run app.py
Expects the following alongside this file (produced by WorkBalanceAI_Analysis.ipynb):
  models/scaler.pkl, models/pca.pkl, models/mlp_classifier.pkl, models/shap_rf.pkl
  models/cluster_name_map.json, models/feature_list.json
  app_data/country_summary.csv, app_data/employees_sample.csv

Note: the processed-CSV folder is named "app_data", not "data". On Windows,
folder names are case-insensitive, so "data" would collide with the "Data"
folder that holds the original Data.csv / Codebook.csv files.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

# Resolve paths relative to this file, so the app works regardless of the
# directory it's launched from.
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
APP_DATA_DIR = BASE_DIR / "app_data"

# ------------------------------------------------------------------
# PAGE CONFIG + THEME
# ------------------------------------------------------------------
st.set_page_config(page_title="WorkBalance AI", page_icon="🧭", layout="wide")

PALETTE = {
    "bg": "#0f1720",
    "panel": "#16212c",
    "panel_alt": "#1c2a37",
    "text": "#e8edf2",
    "muted": "#8fa1b3",
    "accent": "#e8a13a",   # warm amber — the "signal" color, used sparingly
    "healthy": "#4fb286",
    "engaged": "#e8c548",
    "overworked": "#e8853a",
    "addicted": "#d9534f",
}
PROFILE_COLORS = {
    "Healthy & Balanced": PALETTE["healthy"],
    "Engaged but Strained": PALETTE["engaged"],
    "Overworked": PALETTE["overworked"],
    "Work-Addicted": PALETTE["addicted"],
}

st.markdown(f"""
<style>
    .stApp {{ background-color: {PALETTE['bg']}; color: {PALETTE['text']}; }}
    section[data-testid="stSidebar"] {{ background-color: {PALETTE['panel']}; }}
    h1, h2, h3 {{ font-family: 'Georgia', 'Iowan Old Style', serif; letter-spacing: 0.2px; }}

    /* --- Readability fix -----------------------------------------------
       Streamlit's built-in theme styles labels, captions, and radio/slider
       text with its own (light-theme) colors, which are close to invisible
       against this app's dark background. Force every text element inside
       the app to use the app's own palette instead. */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"],
    .stRadio label, .stRadio div, .stRadio p,
    .stSlider label, .stSlider p,
    .stSelectSlider label, .stSelectSlider p,
    .stSelectbox label, .stSelectbox p,
    .stMultiSelect label, .stMultiSelect p,
    .stTextInput label, .stTextInput p {{
        color: {PALETTE['text']} !important;
    }}

    /* Captions (st.caption) use a slightly muted but still legible tone */
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
    .stApp small {{
        color: {PALETTE['muted']} !important;
    }}

    /* Sidebar: same treatment, including the nav radio labels */
    section[data-testid="stSidebar"] * {{ color: {PALETTE['text']} !important; }}
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color: {PALETTE['muted']} !important;
    }}

    /* Info/warning boxes: keep Streamlit's icon colors but force dark,
       legible body text since these render on a light chip in some themes */
    [data-testid="stAlert"] p {{ color: {PALETTE['text']} !important; }}

    .wb-card {{
        background-color: {PALETTE['panel']};
        border: 1px solid {PALETTE['panel_alt']};
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
    }}
    .wb-eyebrow {{
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.72rem;
        color: {PALETTE['muted']} !important;
    }}
    .wb-bigstat {{ font-size: 2.4rem; font-weight: 700; color: {PALETTE['text']} !important; }}

    /* Re-affirmed *after* the broad rules above so it always wins the tie */
    .stButton>button {{
        background-color: {PALETTE['accent']} !important;
        color: #1a1408 !important;
        border: none;
        font-weight: 600;
        border-radius: 6px;
    }}
    .stButton>button:hover {{ background-color: #f2b658 !important; color: #1a1408 !important; }}
    .stButton>button p {{ color: #1a1408 !important; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# LOAD ARTIFACTS (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    pca = joblib.load(MODELS_DIR / "pca.pkl")
    mlp = joblib.load(MODELS_DIR / "mlp_classifier.pkl")
    shap_rf = joblib.load(MODELS_DIR / "shap_rf.pkl")
    cluster_map = json.load(open(MODELS_DIR / "cluster_name_map.json"))
    feat_info = json.load(open(MODELS_DIR / "feature_list.json"))
    explainer = shap.TreeExplainer(shap_rf)
    return scaler, pca, mlp, shap_rf, cluster_map, feat_info, explainer


@st.cache_data
def load_data():
    country_df = pd.read_csv(APP_DATA_DIR / "country_summary.csv")
    sample_df = pd.read_csv(APP_DATA_DIR / "employees_sample.csv")
    return country_df, sample_df


scaler, pca, mlp, shap_rf, cluster_map, feat_info, explainer = load_artifacts()
country_df, sample_df = load_data()

FEATURES = feat_info["features"]
IWAS_COLS = feat_info["iwas_cols"]
NAMES_BY_RANK = cluster_map["names_by_rank"]
EMOJI = cluster_map["emoji_by_rank"]

IWAS_QUESTIONS = {
    "IWAS1": "Thought of how you could free up more time to work",
    "IWAS2": "Spent much more time working than initially intended",
    "IWAS3": "Worked to reduce feelings of guilt, anxiety, helplessness or depression",
    "IWAS4": "Been told by others to cut down on work — without listening",
    "IWAS5": "Become stressed if prohibited from working",
    "IWAS6": "Deprioritized hobbies, leisure or exercise because of work",
    "IWAS7": "Worked so much it negatively influenced your health",
    "IWAS8": "Felt work was more important than family or friends",
    "IWAS9": "Felt your entire life was only focused on work",
    "IWAS10": "Been unable to stop thinking about work",
    "IWAS11": "Worked so much it negatively influenced your sleep",
    "IWAS12": "Felt you should work more and more",
    "IWAS13": "Worked to forget about personal problems",
    "IWAS14": "Had serious life problems because of how much you worked",
    "IWAS15": "Tried to reduce your work but failed",
    "IWAS16": "Neglected everything except work",
}
SCALE_5 = {1: "Never", 2: "Rarely", 3: "Sometimes", 4: "Often", 5: "Always"}

# Short, chart-friendly labels for every model feature. The IWAS items reuse
# the full questionnaire wording (IWAS_QUESTIONS) as hover text, so a user
# looking at the "Why this result?" chart never has to decode a code like
# "IWAS16" — they see a readable phrase, and can hover for the exact
# question it came from.
FEATURE_SHORT_LABELS = {
    "IWAS1": "Planning more time to work",
    "IWAS2": "Working longer than planned",
    "IWAS3": "Working to numb negative feelings",
    "IWAS4": "Ignoring requests to cut down",
    "IWAS5": "Stressed when unable to work",
    "IWAS6": "Deprioritizing hobbies for work",
    "IWAS7": "Work harming health",
    "IWAS8": "Work over family & friends",
    "IWAS9": "Life focused only on work",
    "IWAS10": "Can't stop thinking about work",
    "IWAS11": "Work harming sleep",
    "IWAS12": "Feeling pressure to work more",
    "IWAS13": "Working to forget problems",
    "IWAS14": "Serious life problems from work",
    "IWAS15": "Failed attempts to cut back",
    "IWAS16": "Neglecting everything but work",
    "JStress": "Job stress",
    "JSatisf": "Job satisfaction",
    "SEsteem": "Self-esteem",
}
# Full source text for the hover tooltip (falls back to the short label for
# the three non-questionnaire features, which are already self-explanatory).
FEATURE_FULL_TEXT = {
    **IWAS_QUESTIONS,
    "JStress": "Job stress: how stressful is your job right now?",
    "JSatisf": "Job satisfaction: how satisfied are you with your job right now?",
    "SEsteem": "Self-esteem: how satisfied are you with yourself right now?",
}


def predict_profile(answers: dict):
    """answers: dict of FEATURES -> value. Returns profile name, probs dict, x_scaled."""
    x = np.array([[answers[f] for f in FEATURES]])
    x_scaled = scaler.transform(x)
    proba = mlp.predict_proba(x_scaled)[0]
    # mlp classes_ are the raw 'cluster' ints (0..3, 0=healthiest per our ranking in the notebook)
    classes = mlp.classes_
    probs_by_name = {}
    for cls, p in zip(classes, proba):
        probs_by_name[NAMES_BY_RANK[cls]] = p
    pred_cluster = classes[int(np.argmax(proba))]
    pred_name = NAMES_BY_RANK[pred_cluster]
    return pred_name, probs_by_name, x_scaled


# ------------------------------------------------------------------
# SIDEBAR NAV
# ------------------------------------------------------------------
st.sidebar.markdown("## 🧭 WorkBalance AI")
st.sidebar.caption("AI insights into work stress across 85 cultures")
page = st.sidebar.radio("Go to", ["🧠 Profile Predictor", "🌍 Global Work Map", "🧩 Profile Explorer", "ℹ️ About"])
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built on a 31,352-respondent, 85-culture survey of work addiction, "
    "job stress, job satisfaction and self-esteem. Profiles were **discovered** "
    "with clustering, not assumed in advance."
)

# ==================================================================
# PAGE 1 — PROFILE PREDICTOR
# ==================================================================
if page == "🧠 Profile Predictor":
    st.title("🧠 What's your work profile?")
    st.write(
        "Answer the same 19 questions used in the study. A neural network — trained to "
        "reproduce data-driven clusters discovered in 31,352 real responses — will place "
        "you in one of four work-behavior profiles."
    )

    with st.form("questionnaire"):
        st.markdown("#### Over the last year, how often have you...")
        answers = {}
        cols = st.columns(2)
        for i, code in enumerate(IWAS_COLS):
            col = cols[i % 2]
            with col:
                val = st.select_slider(
                    f"**{IWAS_QUESTIONS[code]}**",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    format_func=lambda v: SCALE_5[v],
                    key=code,
                )
                answers[code] = val

        st.markdown("#### A few more about your job right now")
        c1, c2, c3 = st.columns(3)
        with c1:
            answers["JStress"] = st.slider(
                "Job stress (1 = not at all stressful, 7 = extremely stressful)", 1, 7, 4)
        with c2:
            answers["JSatisf"] = st.slider(
                "Job satisfaction (1 = extremely dissatisfied, 7 = extremely satisfied)", 1, 7, 4)
        with c3:
            answers["SEsteem"] = st.slider(
                "Self-esteem (1 = very dissatisfied, 9 = very satisfied)", 1, 9, 5)

        submitted = st.form_submit_button("🔍 Analyze my profile")

    if submitted:
        pred_name, probs_by_name, x_scaled = predict_profile(answers)
        st.session_state["last_prediction"] = {
            "profile": pred_name, "probs": probs_by_name,
            "x_scaled": x_scaled, "answers": answers,
        }

    if "last_prediction" in st.session_state:
        res = st.session_state["last_prediction"]
        pred_name = res["profile"]
        probs_by_name = res["probs"]

        st.markdown("---")
        left, right = st.columns([1, 1.3])

        with left:
            st.markdown(f"""
            <div class="wb-card" style="text-align:center; border-left: 5px solid {PROFILE_COLORS[pred_name]};">
                <div class="wb-eyebrow">Your work profile</div>
                <div style="font-size:3rem;">{EMOJI[pred_name]}</div>
                <div class="wb-bigstat">{pred_name}</div>
                <div style="color:{PALETTE['muted']};">{probs_by_name[pred_name]*100:.0f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

            for name in NAMES_BY_RANK:
                p = probs_by_name.get(name, 0)
                st.markdown(f"**{EMOJI[name]} {name}** — {p*100:.0f}%")
                st.progress(min(max(p, 0.0), 1.0))

        with right:
            st.markdown("##### 🔎 Why this result?")
            st.caption("Top factors pushing your answers toward this profile (SHAP values).")
            classes = list(shap_rf.classes_)
            class_idx = classes.index([c for c in classes if NAMES_BY_RANK[c] == pred_name][0])
            sv = explainer.shap_values(res["x_scaled"])
            sv_row = sv[0, :, class_idx] if np.ndim(sv) == 3 else sv[class_idx][0]

            contrib = pd.DataFrame({
                "Code": FEATURES,
                "Factor": [FEATURE_SHORT_LABELS.get(f, f) for f in FEATURES],
                "FullText": [FEATURE_FULL_TEXT.get(f, f) for f in FEATURES],
                "Impact": sv_row,
            })
            contrib["AbsImpact"] = contrib["Impact"].abs()
            contrib = contrib.sort_values("AbsImpact", ascending=False).head(8)
            contrib["Direction"] = np.where(contrib["Impact"] >= 0, "Pushes toward", "Pushes away from")
            contrib_sorted = contrib[::-1]  # largest impact at the top of the horizontal bar chart

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=contrib_sorted["Factor"], x=contrib_sorted["Impact"], orientation="h",
                marker_color=[PALETTE["addicted"] if v >= 0 else PALETTE["healthy"] for v in contrib_sorted["Impact"]],
                customdata=contrib_sorted[["FullText", "Direction"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]} this profile (impact: %{x:.3f})"
                    "<extra></extra>"
                ),
            ))
            fig.update_layout(
                template="plotly_dark", plot_bgcolor=PALETTE["panel"], paper_bgcolor=PALETTE["panel"],
                height=380, margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title=f"Impact on '{pred_name}' prediction",
                font=dict(color=PALETTE["text"]),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Each bar is one of your questionnaire answers. 🔴 Red bars push the model "
                "toward this profile; 🟢 green bars push away from it — the longer the bar, "
                "the bigger the effect. Hover over a bar to see the exact question it came "
                "from. This reflects patterns learned from the dataset, not a clinical "
                "assessment."
            )

        st.info(
            "This tool is for self-reflection and is based on a large cross-cultural research "
            "dataset — it is **not** a medical or clinical diagnosis of burnout or addiction.",
            icon="ℹ️",
        )

# ==================================================================
# PAGE 2 — GLOBAL WORK MAP
# ==================================================================
elif page == "🌍 Global Work Map":
    st.title("🌍 Global Work Balance")
    st.write(
        "Average work-addiction score by country. **Sample sizes vary a lot (roughly 100–1,500 "
        "respondents per country)** — hover to check `n` before drawing conclusions, and treat this "
        "as descriptive, not a ranking of which culture is 'better'."
    )

    metric = st.selectbox(
        "Color the map by:",
        ["WAS_mean", "JStress_mean", "JSatisf_mean", "SEsteem_mean"],
        format_func=lambda x: {
            "WAS_mean": "Work-addiction score (avg)",
            "JStress_mean": "Job stress (avg)",
            "JSatisf_mean": "Job satisfaction (avg)",
            "SEsteem_mean": "Self-esteem (avg)",
        }[x],
    )
    color_scale = "OrRd" if metric in ["WAS_mean", "JStress_mean"] else "Greens"

    fig = px.choropleth(
        country_df, locations="ISO3", color=metric, hover_name="Country",
        hover_data={"n": True, "ISO3": False, "Top_Profile": True},
        color_continuous_scale=color_scale,
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
        geo=dict(bgcolor=PALETTE["bg"], showframe=False, showcoastlines=False),
        margin=dict(l=0, r=0, t=10, b=0), height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Look up a country")
    country_choice = st.selectbox("Country", sorted(country_df["Country"].unique()))
    row = country_df[country_df["Country"] == country_choice].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", int(row["n"]))
    c2.metric("Work addiction", f"{row['WAS_mean']:.2f} / 5")
    c3.metric("Job stress", f"{row['JStress_mean']:.2f} / 7")
    c4.metric("Job satisfaction", f"{row['JSatisf_mean']:.2f} / 7")
    st.markdown(
        f"Most common profile in **{country_choice}**: "
        f"{EMOJI[row['Top_Profile']]} **{row['Top_Profile']}**"
    )
    if row["n"] < 200:
        st.warning(f"Small sample size (n={int(row['n'])}) — read this country's numbers with caution.")

    st.markdown("---")
    st.markdown("#### Top 15 countries by average work-addiction score")
    top15 = country_df.nlargest(15, "WAS_mean").sort_values("WAS_mean")
    fig2 = px.bar(
        top15, x="WAS_mean", y="Country", orientation="h",
        hover_data=["n"], color="WAS_mean", color_continuous_scale="OrRd",
    )
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"],
        height=480, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ==================================================================
# PAGE 3 — PROFILE EXPLORER
# ==================================================================
elif page == "🧩 Profile Explorer":
    st.title("🧩 Employee Profile Explorer")
    st.write(
        "Each dot is one respondent, projected onto the two dimensions that explain the most "
        "variation in the questionnaire (PCA). Filter by country or profile to see how the "
        "population shifts."
    )

    colf1, colf2 = st.columns(2)
    with colf1:
        countries = st.multiselect(
            "Filter by country (leave empty = all)", sorted(sample_df["Country"].unique())
        )
    with colf2:
        profiles = st.multiselect(
            "Filter by profile (leave empty = all)", NAMES_BY_RANK
        )

    plot_df = sample_df.copy()
    if countries:
        plot_df = plot_df[plot_df["Country"].isin(countries)]
    if profiles:
        plot_df = plot_df[plot_df["Profile"].isin(profiles)]

    fig = px.scatter(
        plot_df, x="PCA1", y="PCA2", color="Profile", color_discrete_map=PROFILE_COLORS,
        category_orders={"Profile": NAMES_BY_RANK},
        hover_data=["Country", "WAS_mean", "JStress", "JSatisf"],
        opacity=0.55,
    )

    # If the user has a prediction from the Predictor page, show where they land
    if "last_prediction" in st.session_state:
        res = st.session_state["last_prediction"]
        user_pca = pca.transform(res["x_scaled"])
        fig.add_trace(go.Scatter(
            x=[user_pca[0, 0]], y=[user_pca[0, 1]], mode="markers",
            marker=dict(size=18, color=PALETTE["accent"], symbol="star", line=dict(width=2, color="white")),
            name="You",
        ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["panel"],
        height=560, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="PCA 1 (general work-strain axis)", yaxis_title="PCA 2",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Showing {len(plot_df):,} respondents (down-sampled for speed, ~120 per country).")

    if "last_prediction" not in st.session_state:
        st.info("Tip: complete the **Profile Predictor** page first to see your own ⭐ position on this map.")

# ==================================================================
# PAGE 4 — ABOUT
# ==================================================================
else:
    st.title("ℹ️ About WorkBalance AI")
    st.markdown("""
This app is the deployment layer for a data science project built on a cross-cultural
survey of **31,352 respondents across 85 cultures**, measuring:

- 16 items from a work-addiction questionnaire (last 12 months)
- Job stress, job satisfaction (single items)
- Self-esteem (single item)

**Method, in short**
1. Data cleaning (recoding missing-value codes, median imputation for a small % of missing values).
2. PCA + K-Means to **discover** four data-driven work-behavior profiles: 🟢 Healthy & Balanced,
   🟡 Engaged but Strained, 🟠 Overworked, 🔴 Work-Addicted.
3. A neural network trained to reproduce those cluster assignments from the raw questionnaire
   answers, so a *new* respondent can get an instant profile + confidence score.
4. SHAP explainability to show which answers drove each prediction.
5. This Streamlit app: predictor, world map, and profile explorer.

**What this is not**
- Not a clinical diagnosis of burnout or work addiction.
- Not a claim that any one culture handles work "better" — cross-country comparisons are
  shown with sample sizes and should be read descriptively.
- Not causal — this is cross-sectional self-report survey data.

See the accompanying Jupyter notebook, **WorkBalanceAI_Analysis.ipynb**, for the full analysis,
including EDA, model comparison, and the reasoning behind every choice above.
""")
