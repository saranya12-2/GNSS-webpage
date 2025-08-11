import streamlit as st
from openpyxl import Workbook
import io
import re

print("✅ Running latest version")  # Confirms you're using this file

# --- UI Setup ---
st.set_page_config(page_title="ISMR to Excel Converter", layout="centered")
st.title("📄 ISMR to Excel Converter")

MAX_FILES = 31
uploaded_files = st.file_uploader(
    "📂 Upload up to 31 ISMR files (e.g., 1 month of daily logs)",
    type=["ismr", "txt"],
    accept_multiple_files=True
)

use_header = st.checkbox("Use first non-comment row as headers", value=False)

# --- Validation ---
if uploaded_files:
    if len(uploaded_files) > MAX_FILES:
        st.error(f"🚫 You uploaded {len(uploaded_files)} files. Limit is {MAX_FILES}.")
        uploaded_files = None

# --- Process Files ---
if uploaded_files:
    st.info(f"📅 You uploaded {len(uploaded_files)} file(s). Each will become a sheet in one Excel file.")

    output = io.BytesIO()
    wb = Workbook()
    wb.remove(wb.active)  # Remove default sheet

    for file in uploaded_files:
        st.markdown(f"---\n### 📂 Processing: `{file.name}`")

        try:
            # Read and clean lines
            content = file.read().decode("utf-8", errors="ignore")
            lines = content.strip().splitlines()
            data_lines = [line.strip() for line in lines if line.strip() and not line.startswith("#")]

            if not data_lines:
                st.warning(f"⚠️ `{file.name}` is empty or contains only comments.")
                continue

            # Split each line into list (comma-delimited)
            parsed = [line.split(',') for line in data_lines]
            max_len = max(len(row) for row in parsed)
            normalized = [row + [''] * (max_len - len(row)) for row in parsed]

            # Create sheet
            sheet_name = re.sub(r'[^A-Za-z0-9]', '_', file.name.rsplit('.', 1)[0])[:31]
            ws = wb.create_sheet(title=sheet_name)

            if use_header:
                ws.append(normalized[0])  # header
                for row in normalized[1:]:
                    ws.append(row)
            else:
                for row in normalized:
                    ws.append(row)

            st.success(f"✅ Sheet created: `{sheet_name}` with {len(normalized)} rows.")

        except Exception as e:
            st.error(f"❌ Error processing `{file.name}`: {e}")

    # Save workbook
    wb.save(output)
    output.seek(0)

    st.download_button(
        label="⬇️ Download Excel File",
        data=output,
        file_name="ismr_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
