import streamlit as st
import pandas as pd
import os
import google.generativeai as genai

# --- CONFIGURE STREAMLIT PAGE ---
st.set_page_config(page_title="Negative Keyword Tool", page_icon="🔍")

st.title("🔍 Automated Negative Keyword Manager")

# --- GET GOOGLE GEMINI API KEY FROM ENVIRONMENT VARIABLES ---
gemini_api_key = os.getenv("GEMINI_API_KEY")  # ✅ Fetch API key securely

if not gemini_api_key:
    st.error("❌ Google Gemini API key is missing. Please set GEMINI_API_KEY in your environment.")
    st.stop()

# ✅ Initialize Google Gemini client
genai.configure(api_key=gemini_api_key)

# --- SIDEBAR FOR FILE UPLOAD ---
st.sidebar.header("Upload Google Ads Search Term Report")
uploaded_file = st.sidebar.file_uploader("Upload Google Ads CSV", type=["csv"])

# --- FUNCTION: CLASSIFY NEGATIVE KEYWORDS BASED ON CTR & CONVERSIONS ---
def classify_negative_keywords(data):
    CTR_THRESHOLD = 1.5  # Mark as negative if CTR is below this
    CONVERSION_THRESHOLD = 1  # If conversions are 0, consider negative

    # Convert CTR and Conversions columns to numeric (Fix for TypeError)
    data["CTR (%)"] = pd.to_numeric(data["CTR (%)"], errors="coerce")
    data["Conversions"] = pd.to_numeric(data["Conversions"], errors="coerce")

    # Fill NaN values (if any) with 0
    data["CTR (%)"].fillna(0, inplace=True)
    data["Conversions"].fillna(0, inplace=True)

    # Apply rules
    data["Negative"] = (data["CTR (%)"] < CTR_THRESHOLD) & (data["Conversions"] <= CONVERSION_THRESHOLD)
    return data

# --- FUNCTION: GEMINI NLP ANALYSIS FOR CONTEXT FILTERING ---
def classify_with_gemini(search_term):
    prompt = f"""
    Classify the following search term as 'Relevant' or 'Irrelevant' for a Google Ads campaign:
    Search Term: {search_term}
    - Relevant if it indicates buying intent (buying, subscribing, requesting a demo).
    - Irrelevant if it's job-related, educational, or not related to the business.
    Answer only 'Relevant' or 'Irrelevant'.
    """

    model = genai.GenerativeModel("gemini-pro")  # ✅ Use Gemini Pro Model
    response = model.generate_content(prompt)

    return response.text.strip()

# --- PROCESS UPLOADED CSV FILE ---
if uploaded_file:
    st.success("File uploaded successfully!")
    search_data = pd.read_csv(uploaded_file)

    # Validate expected columns
    required_columns = {"Search Term", "Clicks", "Impressions", "CTR (%)", "Conversions"}
    if not required_columns.issubset(search_data.columns):
        st.error(f"CSV file must contain the following columns: {required_columns}")
        st.stop()

    # Display uploaded data
    st.write("### Uploaded Data Preview:")
    st.write(search_data.head())

    # Apply CTR & Conversion-based filtering
    search_data = classify_negative_keywords(search_data)

    # Apply NLP filtering using Google Gemini
    search_data["NLP Classification"] = search_data["Search Term"].apply(classify_with_gemini)

    # Final filtering: Mark as negative only if CTR & Conversions + NLP classification say "Irrelevant"
    search_data["Final Negative"] = (search_data["Negative"] == True) & (search_data["NLP Classification"] == "Irrelevant")

    # Display flagged negative keywords
    final_negatives = search_data[search_data["Final Negative"] == True]
    st.write("### AI-Flagged Negative Keywords:")
    st.write(final_negatives[["Search Term", "Clicks", "CTR (%)", "Conversions"]])

    # --- ALLOW DOWNLOAD OF NEGATIVE KEYWORDS ---
    def export_negative_keywords(data):
        df = data[["Search Term"]]
        return df.to_csv(index=False).encode("utf-8")

    if not final_negatives.empty:
        st.download_button(
            label="Download Negative Keywords",
            data=export_negative_keywords(final_negatives),
            file_name="negative_keywords.csv",
            mime="text/csv",
        )
