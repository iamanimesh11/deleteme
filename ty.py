import streamlit as st
import gspread
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow

# Authenticate with Google Sheets using Streamlit secrets
def authenticate_google_sheets():
    creds = st.secrets["gcp_service_account"]
    client = gspread.service_account_from_dict(creds)
    return client

# Main function
def main():
    st.title("Rating App")

    # Authenticate with Google Sheets
    gs_client = authenticate_google_sheets()
    if not gs_client:
        st.warning("Authentication failed. Please check your credentials.")
        return

    # Rating input
    rating = st.slider("Rate this app:", min_value=1, max_value=5, value=3)

    if st.button("Submit"):
        # Write rating to Google Sheet
        sheet = gs_client.open("hello").sheet1
        sheet.append_row([rating])
        st.success("Rating submitted!")

if __name__ == "__main__":
    main()
