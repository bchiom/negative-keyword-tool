import streamlit as st
import pandas as pd
import os
from openai import OpenAI
import time

# --- CONFIGURE STREAMLIT PAGE ---
st.set_page_config(page_title="Negative Keyword Tool", page_icon="🔍")

st.title("🔍 Automated Negative Keyword Manager")

# --- GET OPENAI API KEY FROM ENVIRONMENT VARIABLES ---
openai_api_key = os.getenv("OPENAI_API_KEY")  # ✅ Fetch API key securely

if not openai_api_key:
    st.error("❌ OpenAI API key is missing. Please set OPENAI_API_KEY in your environment.")
    st.stop()

# ✅ Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# --- SIDEBAR FOR FILE UPLOAD ---
st.sidebar.header("Upload Google Ads Search Term Report")
uploaded_file = st.sidebar.file_uploader("Upload Google Ads CSV", type=["csv"])

# --- FUNCTION: CLASSIFY NEGATIVE KEYWORDS BASED ON CTR & CONVERSIONS ---
def classify_negative_keywords(data):
    CTR_THRESHOLD = 1.5  # Mark as negative if CTR is below this
    CONVERSION_THRESHOLD = 1  # If conversions are 0, consider negative

    data["CTR (%)"] = pd.to_numeric(data["CTR (%)"], errors="coerce").fillna(0)
    data["Conversions"] = pd.to_numeric(data["Conversions"], errors="coerce").fillna(0)

    data["Negative"] = (data["CTR (%)"] < CTR_THRESHOLD) & (data["Conversions"] <= CONVERSION_THRESHOLD)
    return data

# --- FUNCTION: OPENAI GPT ANALYSIS FOR CONTEXT FILTERING WITH RATE LIMITING ---
def classify_with_gpt(search_term):
    prompt = f"""
    Classify the following search term as 'Relevant' or 'Irrelevant' for a Google Ads campaign:
    Search Term: {search_term}
    - Relevant if it indicates buying intent (buying, subscribing, requesting a demo).
    - Irrelevant if it's job-related, educational, or not related to the business.
    Answer only 'Relevant' or 'Irrelevant'.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        classification = response.choices[0].message.content.strip()
        if classification not in ["Relevant", "Irrelevant"]:
            return "Unknown"
        time.sleep(60)  # Rate limiting to avoid API quota issues
        return classification
    except Exception as e:
        st.error(f"API Error for term '{search_term}': {e}")
        return "Error"

# --- PROCESS UPLOADED CSV FILE ---
if uploaded_file:
    st.success("File uploaded successfully!")
    search_data = pd.read_csv(uploaded_file)

    required_columns = {"Search Term", "Clicks", "Impressions", "CTR (%)", "Conversions"}
    if not required_columns.issubset(search_data.columns):
        st.error(f"CSV file must contain the following columns: {required_columns}")
        st.stop()

    st.write("### Uploaded Data Preview:")
    st.write(search_data.head())

    search_data = classify_negative_keywords(search_data)

    # Apply NLP filtering
    search_data["NLP Classification"] = search_data["Search Term"].apply(classify_with_gpt)

    # Final filtering based on NLP & metrics
    search_data["Final Negative"] = (
        (search_data["Negative"] == True) & (search_data["NLP Classification"] == "Irrelevant")
    )

    # Display flagged negative keywords
    final_negatives = search_data[search_data["Final Negative"] == True]
    st.write("### AI-Flagged Negative Keywords:")
    st.write(final_negatives[["Search Term", "Clicks", "CTR (%)", "Conversions"]])

    # --- DOWNLOAD NEGATIVE KEYWORDS ---
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
