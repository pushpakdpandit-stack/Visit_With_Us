
import streamlit as st
import pandas as pd
import joblib


# Download and load the model
model_path = hf_hub_download(repo_id="Pushpak21/tourism-package-model", filename="best_tourism_package_model.joblib")
model = joblib.load(model_path)


# Feature descriptions
feature_info = {
    "Age": "Age of the customer (years).",
    "TypeofContact": "How the customer was contacted (Company Invited / Self Inquiry).",
    "CityTier": "City category (1=Tier1, 2=Tier2, 3=Tier3).",
    "Occupation": "Customer occupation (Salaried, Freelancer, etc.).",
    "Gender": "Male or Female.",
    "NumberOfPersonVisiting": "Total number of people visiting together.",
    "PreferredPropertyStar": "Preferred hotel star rating (3,4,5).",
    "MaritalStatus": "Single / Married / Divorced.",
    "NumberOfTrips": "Average trips per year.",
    "Passport": "Has passport? (0 = No, 1 = Yes).",
    "OwnCar": "Owns car? (0 = No, 1 = Yes).",
    "NumberOfChildrenVisiting": "Children under 5 accompanying.",
    "Designation": "Job designation/title.",
    "MonthlyIncome": "Gross monthly income.",
    "PitchSatisfactionScore": "Satisfaction score for the sales pitch (1-5).",
    "ProductPitched": "Product variant pitched to the customer.",
    "NumberOfFollowups": "Number of follow-ups by salesperson.",
    "DurationOfPitch": "Duration of pitch in minutes."
}

st.sidebar.title("Feature descriptions")
for k, v in feature_info.items():
    st.sidebar.write(f"**{k}** — {v}")

# Example form using help text (tooltips)
with st.form("input_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30, help=feature_info["Age"])
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"], help=feature_info["TypeofContact"])
    city_tier = st.selectbox("City Tier", [1,2,3], help=feature_info["CityTier"])
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"], help=feature_info["Occupation"])
    gender = st.selectbox("Gender", ["Male", "Female"], help=feature_info["Gender"])
    persons = st.number_input("Number Of Person Visiting", min_value=1, max_value=5, value=2, help=feature_info["NumberOfPersonVisiting"])
    star = st.selectbox("Preferred Property Star", [3,4,5], help=feature_info["PreferredPropertyStar"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced","Unmarried"], help=feature_info["MaritalStatus"])
    trips = st.number_input("Number Of Trips", min_value=1, max_value=25, value=2, help=feature_info["NumberOfTrips"])
    passport = st.radio("Passport", [0,1], help=feature_info["Passport"])
    owncar = st.radio("Own Car", [0,1], help=feature_info["OwnCar"])
    children = st.number_input("Number Of Children Visiting", min_value=0, max_value=3, value=0, help=feature_info["NumberOfChildrenVisiting"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP","VP"], help=feature_info["Designation"])
    income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=30000, help=feature_info["MonthlyIncome"])
    satisfaction = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, help=feature_info["PitchSatisfactionScore"])
    product = st.selectbox("Product Pitched", ["Basic", "Standard","King", "Deluxe", "Super Deluxe"], help=feature_info["ProductPitched"])
    followups = st.number_input("Number Of Followups", min_value=1, max_value=6, value=2, help=feature_info["NumberOfFollowups"])
    duration = st.number_input("Duration Of Pitch (minutes)", min_value=0, max_value=300, value=10, help=feature_info["DurationOfPitch"])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": typeof_contact,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons,
        "PreferredPropertyStar": star,
        "MaritalStatus": marital,
        "NumberOfTrips": trips,
        "Passport": passport,
        "OwnCar": owncar,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income,
        "PitchSatisfactionScore": satisfaction,
        "ProductPitched": product,
        "NumberOfFollowups": followups,
        "DurationOfPitch": duration
    }])
    proba = model.predict_proba(input_df)[0,1]
    pred = model.predict(input_df)[0]
    st.write("Probability:", round(proba,3))
    st.write("Prediction:", "Will buy (1)" if pred==1 else "Will not buy (0)")
