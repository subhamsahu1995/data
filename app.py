import streamlit as st
import pickle

# -------------------------------
# Load Model and Vectorizer
# -------------------------------

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# App Title
# -------------------------------

st.title("🚨 Disaster Tweet Detection App")

st.write("This app predicts whether a tweet about a disaster is **REAL** or **FAKE**.")

st.divider()

# -------------------------------
# User Inputs
# -------------------------------

keyword = st.text_input("Enter Disaster Keyword (Example: earthquake, flood, fire)")

tweet = st.text_area("Enter Tweet Text")

st.divider()

# -------------------------------
# Prediction Button
# -------------------------------

if st.button("Predict"):

    if tweet == "":
        st.warning("Please enter a tweet.")
    
    else:

        # combine keyword + tweet
        input_text = keyword + " " + tweet

        # convert text to vector
        vector = vectorizer.transform([input_text])

        # prediction
        prediction = model.predict(vector)[0]

        # probability
        
        probability = model.predict_proba(vector)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("✅ This is a REAL Disaster Tweet")
        else:
            st.error("❌ This is a FAKE Disaster Tweet")

        st.write("Confidence Score:")

        st.write({
            "Fake Probability": round(probability[0], 3),
            "Real Probability": round(probability[1], 3)
        })

st.divider()

st.caption("Machine Learning Model: Naive Bayes | Text Vectorization: TF-IDF")