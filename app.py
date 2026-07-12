from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score, roc_curve, silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Healthcare Data Mining Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "healthcare_dataset.csv"

PINK, DARK_PINK, SOFT_PINK = "#EC4899", "#BE185D", "#FFF6FA"
PURPLE, GREEN, RED, SLATE = "#7C3AED", "#059669", "#DC2626", "#475569"
COLORS = [PINK, PURPLE, "#2563EB", GREEN, "#F59E0B", RED, "#0EA5E9", "#14B8A6"]

REQUIRED = [
    "Age", "Gender", "Blood Type", "Medical Condition", "Date of Admission",
    "Insurance Provider", "Billing Amount", "Admission Type", "Discharge Date",
    "Medication", "Test Results",
]
MODEL_FEATURES = [
    "Age", "Gender", "Blood Type", "Medical Condition", "Insurance Provider",
    "Admission Type", "Medication", "Test Results", "Length of Stay",
]
CAT_FEATURES = [
    "Gender", "Blood Type", "Medical Condition", "Insurance Provider",
    "Admission Type", "Medication", "Test Results",
]
NUM_FEATURES = ["Age", "Length of Stay"]
CLUSTER_FEATURES = ["Age", "Length of Stay", "Billing Amount"]

st.markdown(
    f"""
    <style>
    .stApp{{background:{SOFT_PINK}}}
    [data-testid="stSidebar"]{{
        background:#FFFFFF;
        border-right:1px solid #FBCFE8
    }}
    [data-testid="stMetric"]{{
        background:#FFF;border:1px solid #FBCFE8;border-radius:18px;
        padding:14px 16px;box-shadow:0 8px 25px rgba(190,24,93,.07)
    }}
    [data-testid="stMetricLabel"]{{color:{SLATE}}}
    [data-testid="stMetricValue"]{{color:{DARK_PINK}}}
    div[data-testid="stVerticalBlockBorderWrapper"]{{
        border-color:#FBCFE8!important;border-radius:18px!important;
        background:#FFF;box-shadow:0 8px 25px rgba(190,24,93,.05)
    }}
    .hero{{
        padding:24px 26px;border-radius:24px;
        background:linear-gradient(120deg,#BE185D 0%,#EC4899 48%,#7C3AED 100%);
        color:#FFF;margin-bottom:18px;box-shadow:0 18px 40px rgba(190,24,93,.20)
    }}
    .hero h1{{margin:0 0 6px;font-size:2rem;color:#FFF}}
    .hero p{{margin:0;opacity:.95;font-size:1rem}}
    .section-title{{font-size:1.12rem;font-weight:750;color:#831843;margin:8px 0}}
    .insight,.recommendation,.warning-note{{
        border-radius:12px;padding:13px 15px;margin:6px 0
    }}
    .insight{{background:#EFF6FF;border-left:5px solid #2563EB}}
    .recommendation{{background:#F0FDF4;border-left:5px solid #16A34A}}
    .warning-note{{background:#FFFBEB;border-left:5px solid #F59E0B}}
    .stButton>button,.stDownloadButton>button{{
        border-radius:12px;border:1px solid {PINK}
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarContent"]{{
        padding-left:.45rem!important;
        padding-right:.45rem!important;
        padding-top:.80rem!important
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]{{
        gap:.22rem!important
    }}
    [data-testid="stSidebar"] .stButton{{
        width:100%!important;
        margin:0!important;
        padding:0!important
    }}
    [data-testid="stSidebar"] .stButton > button{{
        width:100%!important;
        display:flex!important;
        align-items:center!important;
        justify-content:flex-start!important;
        text-align:left!important;
        padding:.78rem .95rem!important;
        margin:0 0 .38rem 0!important;
        background:#FDF2F8!important;
        color:#9D174D!important;
        border:1px solid #FBCFE8!important;
        border-radius:14px!important;
        box-shadow:none!important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover{{
        background:#FCE7F3!important;
        color:#831843!important;
        border-color:#EC4899!important;
    }}
    [data-testid="stSidebar"] .stButton > button:focus{{
        box-shadow:none!important;
        border-color:#EC4899!important;
    }}
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button span,
    [data-testid="stSidebar"] .stButton > button div{{
        width:100%!important;
        margin:0!important;
        text-align:left!important;
        justify-content:flex-start!important;
        align-items:center!important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

def section(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

def usd(value: float) -> str:
    return "$0" if pd.isna(value) else f"${value:,.2f}"

def chart(title: str, fig: go.Figure, height: int = 390) -> None:
    fig.update_layout(
        height=height, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color=SLATE), title_text="", legend_title_text="",
        margin=dict(l=20, r=20, t=20, b=35),
    )
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.plotly_chart(fig, use_container_width=True)

def block(css: str, title: str, items: List[str]) -> None:
    rows = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f'<div class="{css}"><strong>{title}</strong><ul>{rows}</ul></div>',
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def load_data(path: str, mtime: float) -> pd.DataFrame:
    del mtime
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Billing Amount"] = pd.to_numeric(df["Billing Amount"], errors="coerce")
    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
    df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")
    df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
    df["Admission Year"] = df["Date of Admission"].dt.year
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str).str.strip()
    df = df.dropna(subset=MODEL_FEATURES + ["Billing Amount", "Admission Year"])
    df = df[(df["Billing Amount"] > 0) & (df["Length of Stay"] > 0)].copy()
    df["Age"] = df["Age"].astype(int)
    df["Length of Stay"] = df["Length of Stay"].astype(int)
    df["Admission Year"] = df["Admission Year"].astype(int)
    return df.reset_index(drop=True)

def get_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        st.error("File healthcare_dataset.csv tidak ditemukan.")
        st.stop()
    try:
        return load_data(str(DATA_FILE), os.path.getmtime(DATA_FILE))
    except Exception as exc:
        st.error("Dataset gagal dibaca.")
        st.exception(exc)
        st.stop()

def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ],
        remainder="drop",
    )

def clean_name(name: str) -> str:
    replacements = {
        "cat__": "", "num__": "", "Medical Condition_": "Kondisi: ",
        "Insurance Provider_": "Asuransi: ", "Admission Type_": "Admisi: ",
        "Medication_": "Obat: ", "Test Results_": "Tes: ",
        "Blood Type_": "Gol. darah: ", "Gender_": "Gender: ",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name

def feature_importance(model: Pipeline) -> pd.DataFrame:
    names = model.named_steps["prep"].get_feature_names_out()
    values = model.named_steps["model"].feature_importances_
    return pd.DataFrame(
        {"Fitur": [clean_name(x) for x in names], "Importance": values}
    ).sort_values("Importance", ascending=False)

@st.cache_resource(show_spinner=False)
def regression_model(path: str, mtime: float) -> Dict[str, object]:
    df = load_data(path, mtime).sample(min(8000, len(load_data(path, mtime))), random_state=42)
    X, y = df[MODEL_FEATURES], df["Billing Amount"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    model = Pipeline([
        ("prep", preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=45, max_depth=11, min_samples_leaf=3,
            random_state=42, n_jobs=1,
        )),
    ])
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "model": model,
        "metrics": {
            "MAE": mean_absolute_error(yte, pred),
            "RMSE": mean_squared_error(yte, pred) ** .5,
            "R2": r2_score(yte, pred),
        },
        "eval": pd.DataFrame({"Aktual": yte.to_numpy(), "Prediksi": pred}),
        "importance": feature_importance(model),
        "mean": float(y.mean()),
        "q25": float(y.quantile(.25)),
        "q75": float(y.quantile(.75)),
    }

@st.cache_resource(show_spinner=False)
def classification_model(path: str, mtime: float) -> Dict[str, object]:
    full = load_data(path, mtime)
    cost_limit = float(full["Billing Amount"].quantile(.75))
    stay_limit = float(full["Length of Stay"].quantile(.75))
    df = full.sample(min(8000, len(full)), random_state=42)
    target = (
        (df["Billing Amount"] >= cost_limit)
        | (df["Length of Stay"] >= stay_limit)
    ).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        df[MODEL_FEATURES], target, test_size=.2, random_state=42, stratify=target
    )
    model = Pipeline([
        ("prep", preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=60, max_depth=11, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=1,
        )),
    ])
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, prob)
    return {
        "model": model,
        "metrics": {
            "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred, zero_division=0),
            "F1": f1_score(yte, pred, zero_division=0),
            "ROC AUC": roc_auc_score(yte, prob),
        },
        "matrix": confusion_matrix(yte, pred),
        "fpr": fpr,
        "tpr": tpr,
        "importance": feature_importance(model),
    }

def cluster_label(summary: pd.DataFrame, column: str) -> pd.Series:
    return pd.qcut(
        summary[column].rank(method="first"),
        3,
        labels=["Rendah", "Sedang", "Tinggi"],
    )

def cluster_advice(cost: str, stay: str, age: str) -> str:
    if cost == "Tinggi" and stay == "Tinggi":
        text = "Prioritaskan audit biaya, perencanaan tempat tidur, dan koordinasi pembiayaan."
    elif cost == "Tinggi":
        text = "Tinjau tindakan, obat, dan komponen tagihan bernilai tinggi."
    elif stay == "Tinggi":
        text = "Evaluasi hambatan pemulangan dan efisiensi lama rawat."
    elif cost == "Rendah" and stay == "Rendah":
        text = "Pertahankan alur layanan standar yang efisien."
    else:
        text = "Lakukan pemantauan berkala sesuai kebutuhan kelompok."
    if age == "Tinggi":
        text += " Tambahkan perhatian pada kebutuhan pasien usia lanjut."
    return text

@st.cache_data(show_spinner=False)
def clustering_result(path: str, mtime: float, k: int) -> Dict[str, object]:
    df = load_data(path, mtime).copy()
    scaled = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    rng = np.random.default_rng(42)
    fit_idx = rng.choice(len(df), min(20000, len(df)), replace=False)
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled[fit_idx])
    labels = model.predict(scaled)
    df["Cluster"] = labels
    pca = PCA(n_components=2, random_state=42)
    comp = pca.fit_transform(scaled)
    df["PCA 1"], df["PCA 2"] = comp[:, 0], comp[:, 1]
    summary = (
        df.groupby("Cluster")
        .agg(
            Jumlah_Pasien=("Cluster", "size"),
            Rata_Usia=("Age", "mean"),
            Rata_Lama_Rawat=("Length of Stay", "mean"),
            Rata_Biaya=("Billing Amount", "mean"),
        )
        .reset_index()
    )
    summary["Persentase"] = summary["Jumlah_Pasien"] / len(df) * 100
    top_condition = (
        df.groupby("Cluster")["Medical Condition"]
        .agg(lambda s: s.value_counts().index[0])
        .rename("Kondisi_Dominan")
        .reset_index()
    )
    top_admission = (
        df.groupby("Cluster")["Admission Type"]
        .agg(lambda s: s.value_counts().index[0])
        .rename("Admisi_Dominan")
        .reset_index()
    )
    summary = summary.merge(top_condition, on="Cluster").merge(top_admission, on="Cluster")
    summary["Biaya"] = cluster_label(summary, "Rata_Biaya")
    summary["Rawat"] = cluster_label(summary, "Rata_Lama_Rawat")
    summary["Usia"] = cluster_label(summary, "Rata_Usia")
    summary["Profil"] = summary.apply(
        lambda r: f"Biaya {r['Biaya']} - Rawat {r['Rawat']} - Usia {r['Usia']}",
        axis=1,
    )
    summary["Rekomendasi"] = summary.apply(
        lambda r: cluster_advice(str(r["Biaya"]), str(r["Rawat"]), str(r["Usia"])),
        axis=1,
    )
    df["Profil Cluster"] = df["Cluster"].map(summary.set_index("Cluster")["Profil"])
    score = silhouette_score(
        scaled, labels, sample_size=min(5000, len(df)), random_state=42
    )
    return {
        "data": df,
        "summary": summary,
        "silhouette": float(score),
        "variance": float(pca.explained_variance_ratio_.sum()),
    }

def patient_form(df: pd.DataFrame, prefix: str) -> Tuple[pd.DataFrame, bool]:
    with st.form(f"{prefix}_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Usia pasien", 13, 100, 45, key=f"{prefix}_age")
            gender = st.selectbox(
                "Jenis kelamin", sorted(df["Gender"].unique()), key=f"{prefix}_gender"
            )
            blood = st.selectbox(
                "Golongan darah", sorted(df["Blood Type"].unique()), key=f"{prefix}_blood"
            )
        with c2:
            condition = st.selectbox(
                "Kondisi medis",
                sorted(df["Medical Condition"].unique()),
                key=f"{prefix}_condition",
            )
            insurance = st.selectbox(
                "Provider asuransi",
                sorted(df["Insurance Provider"].unique()),
                key=f"{prefix}_insurance",
            )
            admission = st.selectbox(
                "Jenis admisi",
                sorted(df["Admission Type"].unique()),
                key=f"{prefix}_admission",
            )
        with c3:
            medication = st.selectbox(
                "Obat", sorted(df["Medication"].unique()), key=f"{prefix}_medication"
            )
            test = st.selectbox(
                "Hasil tes", sorted(df["Test Results"].unique()), key=f"{prefix}_test"
            )
            stay = st.slider(
                "Lama rawat (hari)", 1, 30, 7, key=f"{prefix}_stay"
            )
        submit = st.form_submit_button("Jalankan prediksi", use_container_width=True)
    row = pd.DataFrame({
        "Age": [age], "Gender": [gender], "Blood Type": [blood],
        "Medical Condition": [condition], "Insurance Provider": [insurance],
        "Admission Type": [admission], "Medication": [medication],
        "Test Results": [test], "Length of Stay": [stay],
    })
    return row, submit

def dashboard_page(df: pd.DataFrame) -> None:
    hero(
        "🏥 Healthcare Analytics Dashboard",
        "Ringkasan data pasien, visualisasi operasional, insight, dan rekomendasi.",
    )
    data = df.copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total pasien", f"{len(data):,}")
    c2.metric("Rata-rata biaya", usd(data["Billing Amount"].mean()))
    c3.metric("Rata-rata lama rawat", f"{data['Length of Stay'].mean():.1f} hari")
    abnormal = data["Test Results"].str.lower().eq("abnormal").mean() * 100
    c4.metric("Hasil tes abnormal", f"{abnormal:.1f}%")

    section("Visualisasi Utama")
    trend = data.groupby("Admission Year").size().reset_index(name="Jumlah Pasien")
    fig = px.line(
        trend, x="Admission Year", y="Jumlah Pasien",
        markers=True, color_discrete_sequence=[PINK],
    )
    fig.update_traces(line=dict(width=3), fill="tozeroy")
    chart("Tren jumlah pasien per tahun", fig)

    left, right = st.columns(2, gap="large")
    with left:
        counts = data["Medical Condition"].value_counts().reset_index()
        counts.columns = ["Kondisi Medis", "Jumlah Pasien"]
        fig = px.bar(
            counts, x="Kondisi Medis", y="Jumlah Pasien", text="Jumlah Pasien",
            color="Kondisi Medis", color_discrete_sequence=COLORS,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        chart("Jumlah pasien berdasarkan kondisi medis", fig)
    with right:
        insurers = data["Insurance Provider"].value_counts().reset_index()
        insurers.columns = ["Provider", "Jumlah Pasien"]
        fig = px.pie(
            insurers, names="Provider", values="Jumlah Pasien",
            hole=.5, color_discrete_sequence=COLORS,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False)
        chart("Proporsi pasien berdasarkan provider asuransi", fig)

    top_condition = data["Medical Condition"].value_counts().idxmax()
    top_insurer = data["Insurance Provider"].value_counts().idxmax()
    top_admission = data["Admission Type"].value_counts().idxmax()
    section("Insight dan Rekomendasi")
    c1, c2 = st.columns(2)
    with c1:
        block("insight", "Insight otomatis", [
            f"Kondisi medis terbanyak adalah {top_condition}.",
            f"Provider asuransi terbanyak adalah {top_insurer}.",
            f"Jenis admisi paling dominan adalah {top_admission}.",
            f"Rata-rata lama rawat {data['Length of Stay'].mean():.1f} hari.",
        ])
    with c2:
        block("recommendation", "Rekomendasi keputusan", [
            f"Prioritaskan kesiapan layanan untuk kondisi {top_condition}.",
            f"Perkuat koordinasi klaim dengan {top_insurer}.",
            "Gunakan tren pasien untuk perencanaan kapasitas tempat tidur.",
            "Pantau kelompok biaya tinggi melalui klasifikasi dan clustering.",
        ])

def regression_page(df: pd.DataFrame) -> None:
    hero(
        "📈 Regresi - Prediksi Biaya Perawatan",
        "Prediksi biaya perawatan pasien berdasarkan karakteristik dan layanan.",
    )
    result = regression_model(str(DATA_FILE), os.path.getmtime(DATA_FILE))
    m = result["metrics"]
    section("Kinerja Model")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", usd(m["MAE"]))
    c2.metric("RMSE", usd(m["RMSE"]))
    c3.metric("R²", f"{m['R2']:.3f}")

    section("Uji Prediksi Pasien")
    row, submitted = patient_form(df, "reg")
    if submitted:
        predicted = float(result["model"].predict(row)[0])
        mean = float(result["mean"])
        category = (
            "Tinggi" if predicted >= result["q75"]
            else "Rendah" if predicted <= result["q25"]
            else "Sedang"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimasi biaya", usd(predicted))
        c2.metric("Rata-rata dataset", usd(mean))
        c3.metric("Kategori biaya", category)
        compare = pd.DataFrame({
            "Kategori": ["Prediksi pasien", "Rata-rata dataset"],
            "Billing Amount": [predicted, mean],
        })
        fig = px.bar(
            compare, x="Kategori", y="Billing Amount", color="Kategori",
            text=compare["Billing Amount"].map(lambda x: f"${x:,.0f}"),
            color_discrete_sequence=[PINK, PURPLE],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        chart("Perbandingan prediksi dengan rata-rata dataset", fig, 350)
        diff = (predicted - mean) / mean * 100
        advice = {
            "Tinggi": [
                "Siapkan anggaran lebih besar sejak awal.",
                "Verifikasi manfaat asuransi lebih awal.",
                "Pantau komponen biaya selama perawatan.",
            ],
            "Sedang": [
                "Gunakan estimasi sebagai batas awal anggaran.",
                "Pantau perubahan lama rawat dan penggunaan obat.",
            ],
            "Rendah": [
                "Pertahankan alur layanan yang efisien.",
                "Tetap lakukan validasi tagihan.",
            ],
        }[category]
        c1, c2 = st.columns(2)
        with c1:
            block("insight", "Insight prediksi", [
                f"Prediksi biaya berada {abs(diff):.1f}% "
                f"{'di atas' if diff >= 0 else 'di bawah'} rata-rata dataset.",
                f"Pasien masuk kategori biaya {category.lower()}.",
            ])
        with c2:
            block("recommendation", "Rekomendasi keputusan", advice)

    section("Visualisasi Evaluasi Model")
    sample = result["eval"].sample(min(2000, len(result["eval"])), random_state=42)
    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            sample, x="Aktual", y="Prediksi",
            opacity=.45, color_discrete_sequence=[PINK],
        )
        low = min(sample["Aktual"].min(), sample["Prediksi"].min())
        high = max(sample["Aktual"].max(), sample["Prediksi"].max())
        fig.add_trace(go.Scatter(
            x=[low, high], y=[low, high], mode="lines", name="Prediksi ideal",
            line=dict(color=GREEN, dash="dash"),
        ))
        chart("Aktual vs prediksi", fig)
    with right:
        top = result["importance"].head(12).sort_values("Importance")
        fig = px.bar(
            top, x="Importance", y="Fitur", orientation="h",
            color="Importance", color_continuous_scale="RdPu",
        )
        fig.update_layout(coloraxis_showscale=False)
        chart("Fitur paling berpengaruh", fig)

def classification_page(df: pd.DataFrame) -> None:
    hero(
        "🎯 Klasifikasi - Risiko Sumber Daya Tinggi",
        "Klasifikasi pasien yang berpotensi membutuhkan biaya tinggi atau masa rawat panjang.",
    )
    result = classification_model(str(DATA_FILE), os.path.getmtime(DATA_FILE))
    m = result["metrics"]
    section("Kinerja Model")
    cols = st.columns(5)
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    formats = [
        f"{m['Accuracy']:.1%}", f"{m['Precision']:.1%}",
        f"{m['Recall']:.1%}", f"{m['F1']:.1%}", f"{m['ROC AUC']:.3f}",
    ]
    for col, label, value in zip(cols, labels, formats):
        col.metric(label, value)

    section("Uji Klasifikasi Pasien")
    row, submitted = patient_form(df, "cls")
    if submitted:
        probability = float(result["model"].predict_proba(row)[0, 1])
        label = (
            "Risiko Sumber Daya Tinggi"
            if probability >= .5 else "Risiko Sumber Daya Normal"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Hasil klasifikasi", label)
        c2.metric("Probabilitas risiko tinggi", f"{probability:.1%}")
        c3.metric("Batas keputusan", "50%")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Probabilitas Risiko Sumber Daya Tinggi"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": PINK},
                "steps": [
                    {"range": [0, 40], "color": "#DCFCE7"},
                    {"range": [40, 70], "color": "#FEF3C7"},
                    {"range": [70, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": RED, "width": 4},
                    "thickness": .8, "value": 50,
                },
            },
        ))
        chart("Skor probabilitas", gauge, 330)
        if probability >= .70:
            insight = "Pasien berada pada prioritas tinggi untuk pemantauan biaya dan kapasitas."
            advice = [
                "Lakukan verifikasi asuransi lebih awal.",
                "Siapkan pengawasan biaya harian.",
                "Koordinasikan kebutuhan tempat tidur sejak awal.",
            ]
        elif probability >= .40:
            insight = "Risiko berada pada tingkat menengah dan perlu pemantauan."
            advice = [
                "Pantau perubahan lama rawat.",
                "Perbarui estimasi biaya saat kondisi berubah.",
            ]
        else:
            insight = "Risiko kebutuhan sumber daya tinggi relatif rendah."
            advice = [
                "Gunakan alur layanan standar.",
                "Tetap lakukan pemeriksaan tagihan.",
            ]
        c1, c2 = st.columns(2)
        with c1:
            block("insight", "Insight klasifikasi", [
                insight, f"Probabilitas hasil model adalah {probability:.1%}."
            ])
        with c2:
            block("recommendation", "Rekomendasi keputusan", advice)

    section("Evaluasi Klasifikasi")
    left, right = st.columns(2)
    with left:
        matrix = pd.DataFrame(
            result["matrix"],
            index=["Aktual: Tidak Tinggi", "Aktual: Tinggi"],
            columns=["Prediksi: Tidak Tinggi", "Prediksi: Tinggi"],
        )
        fig = px.imshow(
            matrix, text_auto=True, color_continuous_scale="RdPu", aspect="auto"
        )
        chart("Confusion matrix", fig)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result["fpr"], y=result["tpr"], mode="lines",
            name=f"ROC (AUC={m['ROC AUC']:.3f})",
            line=dict(color=PINK, width=3),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Acak",
            line=dict(color=SLATE, dash="dash"),
        ))
        fig.update_xaxes(title="False Positive Rate")
        fig.update_yaxes(title="True Positive Rate")
        chart("Kurva ROC", fig)
    top = result["importance"].head(15).sort_values("Importance")
    fig = px.bar(
        top, x="Importance", y="Fitur", orientation="h",
        color="Importance", color_continuous_scale="RdPu",
    )
    fig.update_layout(coloraxis_showscale=False)
    chart("Fitur paling berpengaruh pada klasifikasi", fig, 480)

def clustering_page(df: pd.DataFrame) -> None:
    hero(
        "🧩 Clustering - Segmentasi Pasien",
        "Segmentasi pasien berdasarkan usia, lama rawat, dan biaya perawatan.",
    )
    k = st.slider("Pilih jumlah cluster", 2, 6, 3)
    result = clustering_result(str(DATA_FILE), os.path.getmtime(DATA_FILE), k)
    clustered, summary = result["data"], result["summary"].copy()

    section("Kualitas dan Komposisi Cluster")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah cluster", k)
    c2.metric("Silhouette score", f"{result['silhouette']:.3f}")
    c3.metric("Pasien dianalisis", f"{len(clustered):,}")
    c4.metric("Variansi PCA 2D", f"{result['variance']:.1%}")
    section("Visualisasi Segmentasi")
    sample = clustered.sample(min(5000, len(clustered)), random_state=42)
    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            sample, x="PCA 1", y="PCA 2", color="Profil Cluster",
            hover_data=["Age", "Length of Stay", "Billing Amount", "Medical Condition"],
            opacity=.60, color_discrete_sequence=COLORS,
        )
        chart("Peta cluster 2D menggunakan PCA", fig, 470)
    with right:
        sample3d = sample.sample(min(2500, len(sample)), random_state=7)
        fig = px.scatter_3d(
            sample3d, x="Age", y="Length of Stay", z="Billing Amount",
            color="Profil Cluster",
            hover_data=["Medical Condition", "Admission Type"],
            opacity=.55, color_discrete_sequence=COLORS,
        )
        fig.update_scenes(
            xaxis_title="Usia", yaxis_title="Lama Rawat",
            zaxis_title="Billing Amount",
        )
        chart("Sebaran cluster pada tiga variabel asli", fig, 470)

    section("Profil Tiap Cluster")
    display = summary[
        [
            "Cluster", "Profil", "Jumlah_Pasien", "Persentase",
            "Rata_Usia", "Rata_Lama_Rawat", "Rata_Biaya",
            "Kondisi_Dominan", "Admisi_Dominan",
        ]
    ].copy()
    display.columns = [
        "Cluster", "Profil", "Jumlah Pasien", "Persentase (%)",
        "Rata-rata Usia", "Rata-rata Lama Rawat", "Rata-rata Biaya",
        "Kondisi Dominan", "Admisi Dominan",
    ]
    display["Persentase (%)"] = display["Persentase (%)"].round(2)
    display["Rata-rata Usia"] = display["Rata-rata Usia"].round(1)
    display["Rata-rata Lama Rawat"] = display["Rata-rata Lama Rawat"].round(1)
    display["Rata-rata Biaya"] = display["Rata-rata Biaya"].map(usd)
    st.dataframe(display, use_container_width=True, hide_index=True)

    for _, row in summary.sort_values("Cluster").iterrows():
        with st.expander(f"Cluster {int(row['Cluster'])} - {row['Profil']}"):
            c1, c2 = st.columns(2)
            with c1:
                block("insight", "Insight cluster", [
                    f"Mencakup {int(row['Jumlah_Pasien']):,} pasien "
                    f"({row['Persentase']:.1f}%).",
                    f"Rata-rata usia {row['Rata_Usia']:.1f} tahun.",
                    f"Rata-rata lama rawat {row['Rata_Lama_Rawat']:.1f} hari.",
                    f"Rata-rata biaya {usd(row['Rata_Biaya'])}.",
                    f"Kondisi dominan: {row['Kondisi_Dominan']}.",
                ])
            with c2:
                block("recommendation", "Rekomendasi keputusan", [row["Rekomendasi"]])

def main() -> None:
    df = get_data()

    if "menu" not in st.session_state:
        st.session_state.menu = "Dashboard Utama"

    for value in [
        "Dashboard Utama",
        "Regresi Biaya",
        "Klasifikasi Risiko",
        "Clustering Pasien",
    ]:
        if st.sidebar.button(value, key=f"menu_{value}", use_container_width=True):
            st.session_state.menu = value

    menu = st.session_state.menu
    if menu == "Dashboard Utama":
        dashboard_page(df)
    elif menu == "Regresi Biaya":
        regression_page(df)
    elif menu == "Klasifikasi Risiko":
        classification_page(df)
    else:
        clustering_page(df)

if __name__ == "__main__":
    main()