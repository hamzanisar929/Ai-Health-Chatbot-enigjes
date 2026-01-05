import streamlit as st
from chat_ui import chat_interface
from analytics import analytics_tab

st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🧠",
    layout="wide"
)

tabs = st.tabs([
    "💬 AI Health Chat",
    "📊 Data Distribution",
    "🧠 How It Works",
    "⚠️ Disclaimer"
])

with tabs[0]:
    chat_interface()

with tabs[1]:
    analytics_tab()

with tabs[2]:
    st.header("🧠 How This AI Works")
    st.markdown("""
- Uses **Machine Learning Ensemble Models**
- Learns symptom → disease patterns
- Asks **adaptive follow-up questions**
- Predicts disease with confidence
- Visualizes dataset for transparency
""")

with tabs[3]:
    st.warning("""
⚠️ This system is for educational purposes only.
It does NOT replace professional medical advice.
Always consult a doctor.
""")
