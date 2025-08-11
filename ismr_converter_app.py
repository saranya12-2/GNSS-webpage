import streamlit as st
import pandas as pd
import io

# Title and description
st.title("ISMR to Excel Converter")
st.write("Upload one or more `.ismr` files to convert them to `.xlsx` Excel files.")

# Upload section
uploaded_files = st.file_uploader("Choose ISMR files", accept_multiple_files=True, type=["ismr", "txt"])

# Conversion function
def convert_to_excel(file):
    try:
        # Read ISMR file (space/tab-separated)
        df = pd.read_csv(file, delim_whitespace=True, header=None)

        # Save to Excel in memory (no need to write to disk)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Error converting {file.name}: {e}")
        return None

# If files uploaded, convert each and show download buttons
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown(f"### File: {uploaded_file.name}")
        excel_file = convert_to_excel(uploaded_file)
        if excel_file:
            excel_name = uploaded_file.name.replace(".ismr", ".xlsx").replace(".txt", ".xlsx")
            st.download_button(
                label=f"Download {excel_name}",
                data=excel_file,
                file_name=excel_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
