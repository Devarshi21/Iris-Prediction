import streamlit as st
import pandas as pd
import joblib

# Import the model with joblib
model = joblib.load("notebook/model.joblib")

# Species prediction function
def species_predict(sep_len, sep_wid, pet_len, pet_wid):
    d = [
        {
            "sepal_length":sep_len,
            "sepal_width": sep_wid,
            "petal_length": pet_len,
            "petal_width": pet_wid
        }
    ]
    xnew = pd.DataFrame(d)
    pred = model.predict(xnew)[0]
    probs = model.predict_proba(xnew)
    probs_dct = {}
    species = model.classes_
    for s, p in zip(species, probs.flatten()):
        probs_dct[s] = float(p)

    return pred, probs_dct


# Start creating streamlit app
st.set_page_config(page_title="Iris Prediction")

# Adding title to webpage
st.title("Iris End to End Project")
st.subheader("By Devarshi")

# Take user input
sep_len = st.number_input("Sepal Length", min_value=0.00, step=0.01)
sep_wid = st.number_input("Sepal Width", min_value=0.00, step=0.01)
pet_len = st.number_input("Petal Length", min_value=0.00, step=0.01)
pet_wid = st.number_input("Petal Width", min_value=0.00, step=0.01)

# Creating a button
button = st.button("Predict", type="primary")

# IF button is clicked
if button:
    preds, probs = species_predict(sep_len, sep_wid, pet_len, pet_wid)
    st.subheader(f"Predictions : {preds}")
    for s,p in probs.items():
        st.subheader(f"{s} : Probability{p:4f}")
        st.progress(p)
        