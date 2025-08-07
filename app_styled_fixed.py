
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# הגדרות עיצוב
st.set_page_config(page_title="Parkinson's Predictor", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #6c63ff;'>🧠 Parkinson's Disease Prediction App</h1>",
    unsafe_allow_html=True,
)

st.markdown("## 📌 על הפרויקט")
with st.expander("קרא עוד על המערכת והפרויקט"):
    st.markdown(
        """
        מערכת זו מבוססת על למידת מכונה ומשתמשת במודל שאומן על נתוני אמת ממאגר ה־UCI.
        מטרתה לחזות האם אדם חולה בפרקינסון על בסיס מדדים קוליים שונים.

        **כוללת:**
        - השוואת מודלים ואופטימיזציה (GridSearchCV)
        - שמירת המודל הטוב ביותר
        - ניתוח חזותי של התוצאות
        """
    )

# טעינת המודל
model = joblib.load("model.pkl")

st.markdown("## ✏️ הזן נתונים לצורך ניבוי")

# טופס קלט למשתמש
with st.form("prediction_form"):
    st.write("נא למלא את כל השדות")
    MDVP_Fo = st.number_input("MDVP:Fo(Hz)", value=119.992)
    MDVP_Fhi = st.number_input("MDVP:Fhi(Hz)", value=157.302)
    MDVP_Flo = st.number_input("MDVP:Flo(Hz)", value=74.997)
    MDVP_Jitter = st.number_input("MDVP:Jitter(%)", value=0.00784)
    MDVP_Shimmer = st.number_input("MDVP:Shimmer", value=0.04374)
    NHR = st.number_input("NHR", value=0.026)
    HNR = st.number_input("HNR", value=21.033)
    RPDE = st.number_input("RPDE", value=0.414783)
    DFA = st.number_input("DFA", value=0.815285)
    spread1 = st.number_input("spread1", value=-4.813031)
    spread2 = st.number_input("spread2", value=0.266482)
    D2 = st.number_input("D2", value=2.301442)
    PPE = st.number_input("PPE", value=0.284654)

    submitted = st.form_submit_button("בצע ניבוי")

# חיזוי
if submitted:
    expected_columns = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)", "MDVP:Shimmer", "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    input_data = pd.DataFrame([[
        MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Shimmer, NHR, HNR,
        RPDE, DFA, spread1, spread2, D2, PPE
    ]], columns=expected_columns)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"❌ קיים סיכוי גבוה למחלת פרקינסון ({probability:.2%})")
    else:
        st.success(f"✅ לא זוהה סיכון למחלת פרקינסון ({1 - probability:.2%})")

    st.markdown("---")
    st.markdown("### 🔍 ניתוח תוצאה")
    st.write("סיכוי למחלה: {:.2%}".format(probability))

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(x=["Healthy", "Parkinson's"], y=[1 - probability, probability], ax=ax, palette="pastel")
    ax.set_ylabel("Probability")
    st.pyplot(fig)

st.markdown("---")
st.markdown("Created with ❤️ for final ML & AI course project.")
