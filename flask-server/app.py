import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Streamlit title and header
st.title("Attendance Viewer")
st.header("View today's attendance record")

# Get today's date in the format YYYY/MM/DD
date = datetime.now().strftime("%Y/%m/%d")
attendance_dir = "Attendance"
file_name = f"Attendance_{date.replace('/', '-')}.csv"
file_path = os.path.join(attendance_dir, file_name)

# Display the file path being checked
st.write(f"Looking for attendance file: `{file_path}`")

try:
    # Check if the file exists
    if os.path.isfile(file_path):
        st.success(f"Attendance record found for {date}.")

        # Read the attendance CSV
        df = pd.read_csv(file_path)

        # Debugging: Show first few rows (if issues occur)
        st.write("Preview of attendance file (first few rows):")
        st.dataframe(df.head())  # Display the first few rows for confirmation
    else:
        st.warning(f"No attendance record found for {date}.")
except Exception as e:
    st.error(f"An error occurred while loading the attendance: {str(e)}")
