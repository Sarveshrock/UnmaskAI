# examples/show_sample_deepfakes.py  (or wherever you keep it)
import streamlit as st
import pandas as pd
from random import choice  # Better than randrange for lists

# Fixed Google Sheets CSV export link (your sheet is public)
CSV_URL = "https://docs.google.com/spreadsheets/d/1H_kMDdmw3de4BRaf8i7G8x2NgYUM0QkpyTKusb5yC3w/export?format=csv&gid=0"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_examples():
    try:
        df = pd.read_csv(CSV_URL)
        # Assume first column has the video URLs
        return df.iloc[:, 0].dropna().tolist()
    except Exception as e:
        st.error("Could not load examples. Using backup list.")
        # Backup list in case Google Sheets fails
        return [
            "https://www.youtube.com/watch?v=0sR1cr8eX1g",
            "https://www.youtube.com/watch?v=cQ54GDm1eL0",
            "https://www.youtube.com/watch?v=9bZkp7q19f0",
        ]

def examples():
    st.markdown("### 🎥 Sample DeepFake Videos (Click Next for more)")
    
    if "example_urls" not in st.session_state:
        st.session_state.example_urls = load_examples()
    
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0

    urls = st.session_state.example_urls
    if not urls:
        st.error("No example videos loaded.")
        return

    # Show current video
    current_url = urls[st.session_state.current_idx]
    st.video(current_url)

    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("⬅️ Previous"):
            st.session_state.current_idx = (st.session_state.current_idx - 1) % len(urls)
            st.rerun()  # 2025 correct way
    with col2:
        if st.button("Next ➡️"):
            st.session_state.current_idx = (st.session_state.current_idx + 1) % len(urls)
            st.rerun()  # 2025 correct way
    with col3:
        st.write(f"Video {st.session_state.current_idx + 1} of {len(urls)}")

# Call it in your main app like this:
# if add_radio == "Examples":
#     examples()