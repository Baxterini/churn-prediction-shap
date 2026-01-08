import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_LABELS_PL = {
    "CreditScore": "Ocena kredytowa",
    "Geography_Germany": "Kraj: Niemcy",
    "Geography_France": "Kraj: Francja",
    "Geography_Spain": "Kraj: Hiszpania",
    "Gender_Female": "Płeć: Kobieta",
    "Gender_Male": "Płeć: Mężczyzna",
    "Age": "Wiek",
    "Tenure": "Staż (lata)",
    "Balance": "Saldo",
    "NumOfProducts": "Liczba produktów",
    "HasCrCard": "Ma kartę kredytową",
    "IsActiveMember": "Aktywny klient",
    "EstimatedSalary": "Szacowane wynagrodzenie",
}

RAW_LABELS_EN = {k: k.replace("_", " ") for k in RAW_LABELS_PL}

def human_feature_name(feature: str, lang="PL"):
    # num__Age → Age
    # cat__Geography_France → Geography_France
    if "__" in feature:
        base = feature.split("__", 1)[1]
    else:
        base = feature

    if lang == "PL":
        return RAW_LABELS_PL.get(base, base)
    else:
        return RAW_LABELS_EN.get(base, base)

st.set_page_config(page_title="Bank Churn • Explainable AI", layout="wide")

DATA_PATH = Path("data/churn.csv")
MODEL_PATH = Path("models/churn_xgb_pipeline.joblib")

DROP_COLS = ["RowNumber", "CustomerId", "Surname"]
TARGET_COL = "Exited"

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    y = df[TARGET_COL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return df, X_train, X_test, y_train, y_test

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def explain_text_from_waterfall(expl, top_n=5, lang="PL"):
    # expl.values: SHAP contributions for one sample
    vals = np.array(expl.values)
    names = np.array([human_feature_name(f, lang) for f in expl.feature_names])

    order = np.argsort(np.abs(vals))[::-1]
    order = order[:top_n]

    pos = [(names[i], vals[i]) for i in order if vals[i] > 0]
    neg = [(names[i], vals[i]) for i in order if vals[i] < 0]

    if lang == "PL":
        lines = []
        if neg:
            lines.append("**Najmocniej obniża ryzyko churnu:**")
            for n, v in neg[:3]:
                lines.append(f"- {n} ({v:.2f})")
        if pos:
            lines.append("**Najmocniej podnosi ryzyko churnu:**")
            for n, v in pos[:3]:
                lines.append(f"- {n} (+{v:.2f})")
        lines.append("\n💡 **Jak czytać wykres:** czerwone paski zwiększają wynik modelu (ryzyko), niebieskie go zmniejszają. "
                     "Punkt startowy to wartość bazowa, a suma wpływów daje wynik końcowy.")
        return "\n".join(lines)

    # EN
    lines = []
    if neg:
        lines.append("**Strongest churn reducers:**")
        for n, v in neg[:3]:
            lines.append(f"- {n} ({v:.2f})")
    if pos:
        lines.append("**Strongest churn drivers:**")
        for n, v in pos[:3]:
            lines.append(f"- {n} (+{v:.2f})")
    lines.append("\n💡 **How to read it:** red bars push the model output up (higher risk), blue bars push it down. "
                 "The baseline is the starting point; contributions add up to the final output.")
    return "\n".join(lines)

st.title("🏦 Bank Churn • Explainable AI (SHAP)")
st.caption("Wybierz klienta i zobacz predykcję + wyjaśnienie (waterfall).")
st.caption("Select a customer to view prediction and explanation (waterfall).")

pipe = load_pipeline()
df_raw, X_train, X_test, y_train, y_test = load_data()

preprocess = pipe.named_steps["preprocess"]
model = pipe.named_steps["model"]

# Transform test set once (for SHAP + speed)
@st.cache_data
def encode_test(_X_test):
    X_test_enc = preprocess.transform(_X_test)
    feat_names = preprocess.get_feature_names_out()
    return X_test_enc, feat_names

X_test_enc, feature_names = encode_test(X_test)

# Sidebar controls

st.sidebar.header("Ustawienia")
lang = st.sidebar.radio("Język wyjaśnień", ["PL", "EN"], index=0)
threshold = st.sidebar.slider("Próg decyzji (churn = 1)", 0.05, 0.95, 0.50, 0.01)

if lang == "PL":
    st.sidebar.caption(
        "ℹ️ Próg decyzyjny określa, od jakiego prawdopodobieństwa klient "
        "jest uznawany za zagrożonego churnem. "
        "Wyższy próg = mniej alertów, niższy = większa czułość."
    )
else:
    st.sidebar.caption(
        "ℹ️ The decision threshold defines from which probability "
        "a customer is classified as churn risk. "
        "Higher threshold = fewer alerts, lower = higher sensitivity."
    )

idx_list = list(X_test.index)
selected_idx = st.sidebar.selectbox("Wybierz klienta (index z X_test)", idx_list, index=0)
i = idx_list.index(selected_idx)

# Prediction
proba = float(pipe.predict_proba(X_test.loc[[selected_idx]])[:, 1][0])
pred = int(proba >= threshold)

col1, col2, col3 = st.columns(3)
col1.metric("Predykcja", "CHURN (1)" if pred == 1 else "NO CHURN (0)")
col2.metric("P(churn)", f"{proba:.3f}")
col3.metric("Próg", f"{threshold:.2f}")

with st.expander("🔎 Dane klienta (surowe cechy) / 🔎 Customer data (raw features)", expanded=False):
    st.dataframe(X_test.loc[[selected_idx]].T, use_container_width=True)

with st.expander("📘 Słowniczek zmiennych", expanded=False):
    st.markdown("""
**CreditScore** – Ocena kredytowa klienta  
**Geography** – Kraj zamieszkania klienta  
**Gender** – Płeć klienta  
**Age** – Wiek klienta  
**Tenure** – Staż klienta w banku (lata)  
**Balance** – Saldo na koncie  
**NumOfProducts** – Liczba posiadanych produktów bankowych  
**HasCrCard** – Czy klient posiada kartę kredytową  
**IsActiveMember** – Czy klient jest aktywnym użytkownikiem  
**EstimatedSalary** – Szacowane roczne wynagrodzenie
""")

# SHAP explanation for one client
st.subheader("🧠 Wyjaśnienie predykcji (SHAP waterfall)")

x_one = X_test_enc[i:i+1]

# TreeExplainer
explainer = shap.TreeExplainer(model)
sv = explainer(x_one)

# Build Explanation object (handle binary/multiclass shapes)
if len(sv.values.shape) == 3:
    # (n_samples, n_features, n_classes) -> class 1
    explanation = shap.Explanation(
        values=sv.values[0, :, 1],
        base_values=sv.base_values[0, 1],
        data=x_one[0],
        feature_names=feature_names
    )
    base = float(sv.base_values[0, 1])
    fx = base + float(np.sum(sv.values[0, :, 1]))
else:
    explanation = shap.Explanation(
        values=sv.values[0],
        base_values=sv.base_values[0],
        data=x_one[0],
        feature_names=feature_names
    )
    base = float(sv.base_values[0])
    fx = base + float(np.sum(sv.values[0]))

# Plot
fig = plt.figure()
shap.plots.waterfall(explanation, max_display=12, show=False)
st.pyplot(fig, clear_figure=True)

# Optional: show f(x) and approx probability from log-odds
with st.expander("ℹ️ Szczegóły techniczne (opcjonalne)", expanded=False):
    st.write(f"Baseline E[f(X)] = **{base:.3f}**")
    st.write(f"Final f(x) = **{fx:.3f}**")
    st.write(f"Sigmoid(f(x)) ≈ **{sigmoid(fx):.3f}** (przybliżone prawdopodobieństwo z log-odds)")

# Explanation text
if lang == "PL":
    st.markdown("_Uproszczone wyjaśnienie decyzji modelu._")
else:
    st.markdown("_Plain-language explanation of the model decision._")

title = (
    "🗣️ Jak czytać i interpretować wykres"
    if lang == "PL"
    else "🗣️ How to read and interpret the chart"
)

with st.expander(title, expanded=True):

    if lang == "PL":
        st.markdown("""
**Jak czytać wykres SHAP (waterfall):**
- 🔵 **Niebieskie paski** – czynniki, które **zmniejszają ryzyko churnu**
- 🔴 **Czerwone paski** – czynniki, które **zwiększają ryzyko churnu**
- 📏 **Długość paska** – **siła wpływu** danej cechy
- ⚪ **Punkt startowy (baseline)** – średnia predykcja dla wszystkich klientów
- 🎯 **Wynik końcowy** – suma wpływów dla tego konkretnego klienta
""")
    else:
        st.markdown("""
**How to read the SHAP waterfall chart:**
- 🔵 **Blue bars** – factors that **decrease churn risk**
- 🔴 **Red bars** – factors that **increase churn risk**
- 📏 **Bar length** – **strength of the feature’s impact**
- ⚪ **Baseline** – average prediction across all customers
- 🎯 **Final value** – combined effect for this specific customer
""")

    st.markdown("---")
    st.markdown(explain_text_from_waterfall(explanation, top_n=8, lang=lang))


st.caption("Tip: czerwone = podbija wynik modelu, niebieskie = obniża. To lokalne wyjaśnienie dla wybranego klienta.")
