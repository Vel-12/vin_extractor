import os
import cv2
import json
import tempfile
import re
from PIL import Image
import streamlit as st
import pandas as pd
import pypdfium2 as pdfium
from dotenv import load_dotenv
import boto3
from langchain_aws import ChatBedrock
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
AWS_KEY = os.getenv("aws_access_key")
AWS_SECRET = os.getenv("aws_secret_access_key")

client_bedrock = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
    region_name="us-east-1"
)
model = ChatBedrock(
    model="us.meta.llama4-scout-17b-instruct-v1:0",
    client=client_bedrock
)

def calculate_cost(input_tokens, output_tokens):
    return (input_tokens / 1000) * 0.00017 + (output_tokens / 1000) * 0.00066

@st.cache_data
def process_pdf(pdf_path):
    images = []
    doc = pdfium.PdfDocument(pdf_path)
    for i in range(len(doc)):
        page = doc[i]
        pil_image = page.render(scale=2).to_pil().convert('L')
        with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as img_file:
            pil_image.save(img_file.name, format="JPEG")
            images.append(img_file.name)
        page.close()
    doc.close()
    return images

@st.cache_data
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Unreadable image: {path}")
    h, w = img.shape[:2]
    max_dim = 1024
    new_w, new_h = (max_dim, int(h * max_dim / w)) if w > h else (int(w * max_dim / h), max_dim)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    Image.fromarray(resized).save(path, format="JPEG", quality=90)
    return path

def encode_image(path):
    with open(path, "rb") as f:
        return f.read()

def extract_from_image(image_path, client_bedrock):
    payload = encode_image(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"text": "Extract all VIN numbers from this image and return them as a list of strings in JSON format under key 'VINs'."},
            {"image": {"format": "jpeg", "source": {"bytes": payload}}}
        ]
    }]
    response = client_bedrock.converse(
        modelId="us.meta.llama4-scout-17b-instruct-v1:0",
        messages=messages
    )
    text = response["output"]["message"]["content"][0]["text"]
    usage = response["usage"]
    cost = calculate_cost(usage["inputTokens"], usage["outputTokens"])

    json_start, json_end = text.find('{'), text.rfind('}') + 1
    vins = []
    if json_start != -1 and json_end != -1:
        try:
            result = json.loads(text[json_start:json_end])
            vins = result.get("VINs", [])
            if not isinstance(vins, list):
                vins = [vins]
        except:
            pass

    vins += re.findall(r'\b[A-HJ-NPR-Z0-9]{17}\b', text, flags=re.IGNORECASE)
    return list(sorted(set(v.upper() for v in vins))), cost

def process_page_background(img_path, idx, file_name, client_bedrock):
    try:
        preprocess_image(img_path)
        vins, cost = extract_from_image(img_path, client_bedrock)
        return {
            "file": file_name,
            "page": idx + 1,
            "VINs": vins,
            "cost": cost,
            "success": True,
            "image_path": img_path
        }
    except Exception as e:
        return {
            "file": file_name,
            "page": idx + 1,
            "VINs": [],
            "cost": 0,
            "success": False,
            "error_message": f"{file_name} - Page {idx+1}: {e}",
            "image_path": img_path
        }

st.set_page_config("📦 BOL Extractor", layout="wide")
st.markdown("<h1 style='text-align: center;font-size: 42px; color: #4A90E2;'> VIN Extractor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 28px;'>Upload your BOL PDFs or Images</p>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload PDFs or Images", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)

if "slide_index" not in st.session_state:
    st.session_state.slide_index = 0

if uploaded_files:
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        if st.button("🔍 Start Extraction", use_container_width=True):
            all_extracted_data = []
            with st.spinner("Extracting..."):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = []
                    for file in uploaded_files:
                        suffix = file.name.split('.')[-1].lower()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                            tmp.write(file.getvalue())
                            tmp_path = tmp.name

                        if suffix == "pdf":
                            images = process_pdf(tmp_path)
                        else:
                            images = [tmp_path]

                        for idx, img in enumerate(images):
                            futures.append(executor.submit(process_page_background, img, idx, file.name, client_bedrock))

                for future in futures:
                    result = future.result()
                    if result["success"]:
                        all_extracted_data.append(result)
                    else:
                        st.warning(result["error_message"])

            st.session_state.extracted = all_extracted_data
            st.success("✅ Extraction complete!")

if "extracted" in st.session_state:
    all_extracted_data = st.session_state.extracted

    seen_vins = set()
    vin_rows = []
    for item in all_extracted_data:
        if not item["VINs"]:
            vin_rows.append({"File": item["file"], "Page": item["page"], "VIN": "NO VIN"})
        else:
            for vin in item["VINs"]:
                vin = vin.upper()
                if (vin, item["file"], item["page"]) not in seen_vins:
                    seen_vins.add((vin, item["file"], item["page"]))
                    vin_rows.append({
                        "File": item["file"],
                        "Page": item["page"],
                        "VIN": vin
                    })

    df = pd.DataFrame(vin_rows)

    st.divider()
    
    # Create two columns for side-by-side layout
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📋 All VINs Extracted")
        selected_file = st.selectbox("📁 Filter by File", ["All Files"] + df["File"].unique().tolist())
        vin_query = st.text_input("🔎 Search VIN")

        filtered_df = df.copy()
        if selected_file != "All Files":
            filtered_df = filtered_df[filtered_df["File"] == selected_file]
        if vin_query:
            filtered_df = filtered_df[filtered_df["VIN"].str.contains(vin_query.upper(), na=False)]

        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        csv = filtered_df.to_csv(index=False)
        xlsx_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        filtered_df.to_excel(xlsx_tmp.name, index=False)

        export1, export2, export3 = st.columns([1, 8, 1])
        with export1:
            st.download_button("⬇️ CSV", csv, "filtered_vins.csv", mime="text/csv")
        with export3:
            with open(xlsx_tmp.name, "rb") as f:
                st.download_button("⬇️ Excel", f, "filtered_vins.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col_right:
        total = len(all_extracted_data)
        idx = st.session_state.slide_index
        image_info = all_extracted_data[idx]

        st.markdown(f"#### 📄 Preview: {image_info['file']} - Page {image_info['page']}")
        
        # Zoom slider
        zoom_level = st.slider("🔍 Zoom Level", min_value=1.0, max_value=3.0, value=1.0, step=0.1)
        
        # Custom CSS for zoomable image
        st.markdown(
            """
            <style>
            .zoom-container {
                overflow: auto;
                max-height: 600px;
                border: 1px solid #ccc;
                display: flex;
                justify-content: center;
                align-items: center;
                width: 100%;
            }
            .zoom-image {
                transition: transform 0.2s;
                max-width: 100%;
                height: auto;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Display image with zoom
        
            

        nav1, nav2, nav3 = st.columns([1, 3, 1])
        with nav1:
            if st.button("⬅️ Previous Image"):
                st.session_state.slide_index = (idx - 1) % total
        with nav3:
            if st.button("➡️ Next Image"):
                st.session_state.slide_index = (idx + 1) % total
        with st.container():
            st.markdown(f"<div class='zoom-container'>", unsafe_allow_html=True)
            st.image(image_info['image_path'], use_column_width=False, width=int(935 * zoom_level))
            st.markdown("</div>", unsafe_allow_html=True)