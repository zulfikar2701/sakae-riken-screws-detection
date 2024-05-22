import base64
import logging
import uuid
import requests
import boto3
import toml
import streamlit as st
import time
from PIL import Image
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load credentials
# credential = toml.load("credential.toml")
# credential = credential["aws"]
aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

def generate(image_data):
    image_key = f'image/unlabelled/{uuid.uuid4()}.jpg'
    try:
        response = s3_client.generate_presigned_post(Bucket='test-sakaeriken-img-recognition', Key=image_key, ExpiresIn=3600)
    except Exception as e:
        st.error("Presigned URL generation unsuccessful")
        st.error(f"Error: {e}")
        return "Presigned URL generation unsuccessful"

    files = {'file': image_data}  
    r = requests.post(response['url'], data=response['fields'], files=files)
    if r.status_code == 204:
        logging.info(f"Image code: {image_key}")
        
        with st.spinner('Screws Detection Process Is In Progress...'):
            time.sleep(10)
            unique_identifier = image_key.split('/')[-1].split('.')[0]
            labelled_image_key = f'image/labelled/{unique_identifier}.jpg'
        st.success("Result:")
        try:
            content_object = s3_client.get_object(Bucket='test-sakaeriken-img-recognition', Key=labelled_image_key)
            file_content = content_object['Body'].read()
            image = Image.open(BytesIO(file_content))
            st.image(image, caption='Predicted Image', use_column_width=True)
            return "Image retrieved successfully!"

        except Exception as e:
            st.error("Error reading image file from S3")
            st.write(labelled_image_key)
            st.error(f"Error: {e}")
            return "Error reading image file from S3"
    else:
        st.error("Image Uploading Unsuccessful")
        st.error(f"Failed with status code: {r.status_code}")
        return f"Image Uploading Unsuccessful, failed with status code: {r.status_code}"

def main():
    st.title("Sakae Riken Automated Quality Inspection Prototype")
    intro_empty = st.empty()
    intro_empty.subheader("Trained on car bumper screws dataset, upload a picture to detect white and green screws")
    st.sidebar.image("assets/logo.png")
    st.sidebar.title("Automated Quality Inspection")
    option = st.sidebar.selectbox("Choose image to be inspected:", (None, "Take a picture", "Upload a picture"))

    if option == "Take a picture":
        picture = st.sidebar.camera_input("Take a picture")
        if picture:
            st.sidebar.image(picture, caption='Taken Picture', use_column_width=True)
            # Convert to bytes for uploading
            picture_bytes = picture.getvalue()
            intro_empty.empty()
            result = generate(picture_bytes)
            if result == "Image retrieved successfully!":
                logging.info(result)
    
    elif option == "Upload a picture":
        image = st.sidebar.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])
        if image is not None:
            st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)
            # Convert to bytes for uploading
            image_bytes = image.getvalue()
            intro_empty.empty()
            result = generate(image_bytes)
            if result == "Image retrieved successfully!":
                logging.info(result)

if __name__ == "__main__":
    main()
