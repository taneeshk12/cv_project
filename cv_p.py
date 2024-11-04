import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import initialize_app

from PIL import Image
from io import BytesIO

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("../cv_project/path.json")
    initialize_app(cred)

db = firestore.client()

# Load your pre-trained model
model = load_model('../cv_project/license_plate_model.h5')

# Load the cascade classifier for license plate detection
plate_cascade = cv2.CascadeClassifier('../cv_project/indian_license_plate.xml')

# Functions for license plate detection and character segmentation
def detect_plate(img):
    plate_img = img.copy()
    roi = img.copy()
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=7)
    plate = None
    for (x, y, w, h) in plate_rect:
        plate = roi[y:y+h, x:x+w]
        cv2.rectangle(plate_img, (x + 2, y), (x + w - 3, y + h - 5), (51, 181, 155), 3)
    return plate_img, plate

def segment_characters(plate):
    img_lp = cv2.resize(plate, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))
    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]
    dimensions = [LP_WIDTH / 6, LP_WIDTH / 2, LP_HEIGHT / 10, 2 * LP_HEIGHT / 3]
    return find_contours(dimensions, img_binary_lp)

def find_contours(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lower_width, upper_width, lower_height, upper_height = dimensions
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    x_cntr_list = []
    img_res = []
    for cntr in cntrs:
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        if lower_width < intWidth < upper_width and lower_height < intHeight < upper_height:
            x_cntr_list.append(intX)
            char_copy = np.zeros((44, 24))
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))
            char = cv2.subtract(255, char)
            char_copy[2:42, 2:22] = char
            img_res.append(char_copy)
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    return np.array([img_res[idx] for idx in indices])

def show_results(characters):
    dic = {i: c for i, c in enumerate('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
    output = []
    for ch in characters:
        img_ = cv2.resize(ch, (28, 28))  # Ensure the character image is resized to 28x28
        img_ = img_.reshape(28, 28, 1)   # Reshape to (28, 28, 1) for a single channel
        img_ = np.concatenate([img_, img_, img_], axis=-1)  # Convert to (28, 28, 3) by duplicating channels
        img_ = img_.reshape(1, 28, 28, 3)  # Reshape to (1, 28, 28, 3) for model input
        y_pred = model.predict(img_)
        y_ = np.argmax(y_pred, axis=-1)[0]
        output.append(dic[y_])
    return ''.join(output)

# Similarity check function for database lookup
def find_match(predicted_text, threshold=1.0):
    docs = db.collection("Number").get()
    best_match = None
    highest_score = 0
    db_nam = None

    for doc in docs:
        db_text = doc.to_dict().get("Number")  # Replace "Number" with your field name
        if db_text is None:  # Check if db_text is None
            continue  # Skip this iteration if db_text is None
            
        db_name = doc.to_dict().get("Name") 
        # Simple similarity score
        score = sum(1 for a, b in zip(predicted_text, db_text) if a == b) / len(predicted_text)
        if score > highest_score:
            highest_score = score
            best_match = db_text
            db_nam = db_name
    
    if highest_score >= threshold:
        return best_match, highest_score, db_nam
    else:
        return None, 0, None

# Streamlit UI
st.title("License Plate Recognition with Firebase Lookup")

# Create a container for the webcam and image upload
col1, col2 = st.columns(2)

with col1:
    # Start webcam
    video_capture = st.camera_input("Capture a live feed from your webcam", key="webcam_input")

with col2:
    uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"], key="image_upload")

if video_capture is not None:
    # Convert BytesIO to an image
    image = Image.open(video_capture)
    frame = np.array(image)

    # License plate detection
    output_img, plate = detect_plate(frame)
    st.image(output_img, channels="BGR", caption="Detected License Plate", use_column_width=True)

    if plate is not None:
        characters = segment_characters(plate)
        if characters.size > 0:
            predicted_plate_number = show_results(characters)
            st.write("Predicted License Plate Number: ", predicted_plate_number)

            match, score, db_nam = find_match(predicted_plate_number)
            if match:
                st.success(f"Match found in Firebase database with {score * 100:.2f}% similarity: {match} of {db_nam}")
            else:
                st.error("No match found with at least 75% similarity.")
        else:
            st.write("No characters found in the detected license plate.")
    else:
        st.write("No license plate detected.")

elif uploaded_file is not None:
    # Process uploaded image
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_img, plate = detect_plate(img)
    st.image(output_img, channels="BGR", caption="Detected License Plate", use_column_width=True)

    if plate is not None:
        characters = segment_characters(plate)
        if characters.size > 0:
            predicted_plate_number = show_results(characters)
            st.write("Predicted License Plate Number: ", predicted_plate_number)

            match, score, db_nam = find_match(predicted_plate_number)
            if match:
                st.success(f"Match found in Firebase database with {score * 100:.2f}% similarity: {match} of {db_nam}")
            else:
                st.error("No match found with at least 75% similarity.")
        else:
            st.write("No characters found in the detected license plate.")
    else:
        st.write("No license plate detected.")