import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Construction Material AI System",
    layout="wide"
)

st.title("AI-Enabled Construction Material Identification and Volume Estimation")


# -----------------------------
# Folders
# -----------------------------
os.makedirs("input_images", exist_ok=True)
os.makedirs("outputs/preprocessed", exist_ok=True)
os.makedirs("outputs/detected", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)


# -----------------------------
# Model path
# -----------------------------
MODEL_PATH = "runs/detect/train-3/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Check path: runs/detect/train-3/weights/best.pt")
    st.stop()

model = YOLO(MODEL_PATH)


# -----------------------------
# Sidebar settings
# -----------------------------
st.sidebar.header("Detection Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.10,
    0.90,
    0.45,
    0.05
)

bag_volume = st.sidebar.number_input(
    "Volume per Cement Bag in cubic meter",
    min_value=0.01,
    value=0.035,
    step=0.005
)

st.sidebar.info(
    "For multiple angles of the same stack, final count is calculated using median, not sum."
)


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_image(input_path, output_path):
    img = cv2.imread(input_path)

    if img is None:
        return None

    max_width = 900
    h, w = img.shape[:2]

    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    denoised = cv2.bilateralFilter(img, 5, 50, 50)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    l = clahe.apply(l)

    corrected = cv2.merge((l, a, b))
    corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(corrected, (0, 0), 1.0)
    sharpened = cv2.addWeighted(corrected, 1.3, blur, -0.3, 0)

    cv2.imwrite(output_path, sharpened)
    return output_path


# -----------------------------
# False detection filter
# -----------------------------
def is_valid_bag_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return False

    area = bw * bh
    img_area = img_w * img_h
    aspect_ratio = bw / bh

    if area < img_area * 0.003:
        return False

    if area > img_area * 0.25:
        return False

    if aspect_ratio < 1.1 or aspect_ratio > 5.5:
        return False

    if y2 < img_h * 0.18:
        return False

    return True


# -----------------------------
# YOLO detection
# -----------------------------
def detect_cement_bags(image_path, output_path, conf_value):
    img = cv2.imread(image_path)

    if img is None:
        return None, 0, 0

    h, w = img.shape[:2]

    results = model.predict(
        source=image_path,
        conf=conf_value,
        iou=0.35,
        max_det=100,
        save=False,
        verbose=False
    )

    count = 0
    confidences = []

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if not is_valid_bag_box((x1, y1, x2, y2), w, h):
                continue

            count += 1
            confidences.append(conf)

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                img,
                f"cement_bag {conf:.2f}",
                (x1, max(y1 - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0

    cv2.imwrite(output_path, img)

    return output_path, count, avg_conf


# -----------------------------
# CSV Report
# -----------------------------
def generate_csv_report(report_data, final_count, final_volume, count_method):
    csv_path = "outputs/reports/detection_report.csv"

    df = pd.DataFrame(report_data)

    summary = pd.DataFrame([{
        "Image Name": "FINAL RESULT",
        "Detected Bags": final_count,
        "Average Confidence": "-",
        "Estimated Volume": final_volume,
        "Counting Method": count_method
    }])

    df["Counting Method"] = "Individual image count"

    final_df = pd.concat([df, summary], ignore_index=True)
    final_df.to_csv(csv_path, index=False)

    return csv_path


# -----------------------------
# PDF Report
# -----------------------------
def generate_pdf_report(report_data, output_images, final_count, final_volume, count_method):
    pdf_path = "outputs/reports/detection_report.pdf"

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Construction Material Detection Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            styles["Normal"]
        )
    )
    elements.append(Spacer(1, 12))

    table_data = [[
        "Image Name",
        "Detected Bags",
        "Avg Confidence",
        "Estimated Volume"
    ]]

    for row in report_data:
        table_data.append([
            row["Image Name"],
            row["Detected Bags"],
            row["Average Confidence"],
            row["Estimated Volume"]
        ])

    table_data.append([
        "FINAL RESULT",
        final_count,
        count_method,
        final_volume
    ])

    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    for img_path in output_images:
        if os.path.exists(img_path):
            elements.append(Paragraph(os.path.basename(img_path), styles["Heading3"]))
            elements.append(PDFImage(img_path, width=400, height=250))
            elements.append(Spacer(1, 20))

    doc.build(elements)
    return pdf_path


# -----------------------------
# Upload section
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload cement bag images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    report_data = []
    output_images = []
    image_counts = []

    st.warning(
        "If multiple images are different angles of the same stack, final count uses median count, not total sum."
    )

    for index, file in enumerate(uploaded_files, start=1):
        image = Image.open(file).convert("RGB")

        input_path = os.path.join("input_images", file.name)
        image.save(input_path)

        preprocessed_path = os.path.join("outputs/preprocessed", file.name)
        detected_path = os.path.join("outputs/detected", "detected_" + file.name)

        preprocess_result = preprocess_image(input_path, preprocessed_path)

        if preprocess_result is None:
            st.error(f"Could not preprocess {file.name}")
            continue

        detected_output, count, avg_conf = detect_cement_bags(
            preprocessed_path,
            detected_path,
            conf_threshold
        )

        estimated_volume = round(count * bag_volume, 3)

        image_counts.append(count)

        report_data.append({
            "Image Name": file.name,
            "Detected Bags": count,
            "Average Confidence": avg_conf,
            "Estimated Volume": estimated_volume
        })

        output_images.append(detected_path)

        st.subheader(f"Image {index}: {file.name}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                input_path,
                caption="1. Original Image",
                use_container_width=True
            )

        with col2:
            st.image(
                preprocessed_path,
                caption="2. Preprocessed Image",
                use_container_width=True
            )

        with col3:
            st.image(
                detected_path,
                caption=f"3. Detection Output | Count: {count}",
                use_container_width=True
            )

        st.write(f"Detected Bags in this image: **{count}**")
        st.write(f"Average Confidence: **{avg_conf}**")
        st.write(f"Image-wise Estimated Volume: **{estimated_volume} cubic meter**")

        st.divider()

    # -----------------------------
    # Final median logic
    # -----------------------------
    if len(image_counts) == 1:
        final_count = image_counts[0]
        count_method = "Single image count"
    else:
        final_count = int(round(np.median(image_counts)))
        count_method = "Median count from multiple angles"

    final_volume = round(final_count * bag_volume, 3)

    st.header("Final Summary")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Final Cement Bag Count", final_count)

    with colB:
        st.metric("Final Estimated Volume", f"{final_volume} m³")

    with colC:
        st.metric("Counting Method", count_method)

    if len(image_counts) > 1:
        st.subheader("Individual Image Counts")
        for i, count in enumerate(image_counts, start=1):
            st.write(f"Image {i}: {count} bags")

        st.info(
            "Final count is median of image-wise counts. "
            "Counts are not added because the images may show the same stack from different angles."
        )

    # -----------------------------
    # Report generation
    # -----------------------------
    if st.button("Generate CSV and PDF Report"):
        csv_path = generate_csv_report(
            report_data,
            final_count,
            final_volume,
            count_method
        )

        pdf_path = generate_pdf_report(
            report_data,
            output_images,
            final_count,
            final_volume,
            count_method
        )

        with open(csv_path, "rb") as f:
            st.download_button(
                "Download CSV Report",
                f,
                file_name="detection_report.csv",
                mime="text/csv"
            )

        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF Report",
                f,
                file_name="detection_report.pdf",
                mime="application/pdf"
            )

else:
    st.info("Upload cement bag images to start detection.")