import pytesseract
# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import easyocr
import keras_ocr
import pipeline
import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

reader = easyocr.Reader(['en'])

model = YOLO('platmobil.pt')


# Fungsi untuk memprediksi dan menampilkan hasil
def predict_and_plot(image_path):
    # Inisialisasi EasyOCR
    reader = easyocr.Reader(['en'])
    pipeline = keras_ocr.pipeline.Pipeline()

    # YOLO model prediction
    results = model.predict(image_path, device='cpu')

    # Read and process the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]

            # Draw bounding box and label confidence
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Extract Region of Interest (ROI)
            roi = image[y1:y2, x1:x2]

            # EasyOCR
            results_easyocr = reader.readtext(roi)
            easyocr_text = " ".join([result[1] for result in results_easyocr])
            st.write("EasyOCR:", easyocr_text)

            # Tesseract OCR
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            tesseract_text = pytesseract.image_to_string(roi_gray, lang='eng')
            st.write("Tesseract OCR:", tesseract_text.strip())

            # Keras OCR
            prediction_groups = pipeline.recognize([roi])
            keras_ocr_text = " ".join([text for text, box in prediction_groups[0]])
            st.write("Keras OCR:", keras_ocr_text)

    # Display image with bounding boxes
    st.image(image, caption="Predicted Image", use_column_width=True)


# Streamlit UI
st.title("License Plate Detection and OCR")
st.write("Upload an image to detect license plates and extract text using OCR.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to a format readable by OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_path = "temp_image.jpg"
    cv2.imwrite(image_path, image)

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Prediction"):
        predict_and_plot(image_path)