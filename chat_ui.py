import streamlit as st
import time
from chatbot_backend import *

def chat_interface():
    st.title("🧠 AI Health Assistant")

    # ---------------- INITIALIZE SESSION STATE ----------------
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "stage" not in st.session_state:
        st.session_state.stage = "ask_name"
    if "user_data" not in st.session_state:
        st.session_state.user_data = {}
    if "symptoms" not in st.session_state:
        st.session_state.symptoms = []
    if "asked" not in st.session_state:
        st.session_state.asked = []

    # ---------------- DISPLAY CHAT HISTORY ----------------
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ---------------- HELPER FUNCTION ----------------
    def bot_says(text):
        st.session_state.messages.append({"role": "assistant", "content": text})
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(text)
        time.sleep(0.2)

    # ---------------- CHAT INPUT ----------------
    user_input = st.chat_input("Type your message...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        # ---------------- STAGE LOGIC ----------------
        # ASK NAME
        if st.session_state.stage == "ask_name":
            st.session_state.user_data["name"] = user_input
            bot_says(f"Hi {user_input}! How old are you?")
            st.session_state.stage = "ask_age"

        # ASK AGE
        elif st.session_state.stage == "ask_age":
            st.session_state.user_data["age"] = user_input
            bot_says("What is your gender? (M/F/Other)")
            st.session_state.stage = "ask_gender"

        # ASK GENDER
        elif st.session_state.stage == "ask_gender":
            st.session_state.user_data["gender"] = user_input
            bot_says("Great! Now please describe your symptoms in a sentence (e.g., 'I have fever and stomach pain')")
            st.session_state.stage = "symptoms"

        # SYMPTOMS INPUT
        elif st.session_state.stage == "symptoms":
            symptoms = extract_symptoms(user_input, cols)
            if not symptoms:
                bot_says("❌ I couldn’t recognize clear symptoms. Please describe them again.")
            else:
                st.session_state.symptoms.extend(symptoms)
                st.session_state.symptoms = list(set(st.session_state.symptoms))
                bot_says(f"✅ I detected these symptoms: **{', '.join(st.session_state.symptoms)}**")

                # ask follow-up for top predicted disease
                diseases, confidences, _ = predict_disease(st.session_state.symptoms)
                top_disease = diseases[0]
                disease_symptoms = list(
                    training[training['prognosis'] == top_disease].iloc[0][:-1].index[
                        training[training['prognosis'] == top_disease].iloc[0][:-1] == 1
                    ]
                )
                next_q = next((s for s in disease_symptoms if s not in st.session_state.symptoms), None)
                if next_q:
                    st.session_state.asked.append(next_q)
                    bot_says(f"🤔 Do you also have **{next_q.replace('_', ' ')}**?")
                    st.session_state.stage = "followup"
                else:
                    st.session_state.stage = "final"

        # FOLLOW-UP QUESTIONS
        elif st.session_state.stage == "followup":
            answer = user_input.lower()
            last_symptom = st.session_state.asked[-1]
            if answer in ["yes", "y", "yeah"]:
                st.session_state.symptoms.append(last_symptom)

            diseases, confidences, _ = predict_disease(st.session_state.symptoms)
            top_disease = diseases[0]
            disease_symptoms = list(
                training[training['prognosis'] == top_disease].iloc[0][:-1].index[
                    training[training['prognosis'] == top_disease].iloc[0][:-1] == 1
                ]
            )
            next_q = next(
                (s for s in disease_symptoms if s not in st.session_state.symptoms and s not in st.session_state.asked),
                None
            )
            if next_q and len(st.session_state.asked) < 6:
                st.session_state.asked.append(next_q)
                bot_says(f"🤔 Do you also have **{next_q.replace('_', ' ')}**?")
            else:
                st.session_state.stage = "final"

        # FINAL DIAGNOSIS
        if st.session_state.stage == "final":
            diseases, confidences, _ = predict_disease(st.session_state.symptoms)
            top_disease = diseases[0]
            confidence = confidences[0]

            bot_says(f"""
🩺 **Possible Condition:** **{top_disease}**  
🔎 **Confidence:** **{confidence}%**

📖 **About:**  
{description_list.get(top_disease, 'No description available.')}
""")
            if top_disease in precautionDictionary:
                bot_says("🛡️ **Recommended Precautions:**")
                for p in precautionDictionary[top_disease]:
                    bot_says(f"- {p}")
            bot_says("💡 *This is not a medical diagnosis. Please consult a healthcare professional.*")
            st.session_state.stage = "done"
