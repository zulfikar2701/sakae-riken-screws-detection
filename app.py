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
from datetime import datetime
from typing import Optional, Tuple, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'BUCKET_NAME': 'test-sakaeriken-img-recognition',
    'UNLABELLED_PREFIX': 'image/unlabelled/',
    'LABELLED_PREFIX': 'image/labelled/',
    'PRESIGNED_URL_EXPIRY': 3600,
    'MAX_POLLING_ATTEMPTS': 30,
    'POLLING_INTERVAL': 2,  # seconds
    'MAX_RETRIES': 3,
    'MAX_FILE_SIZE_MB': 10,
    'ALLOWED_EXTENSIONS': ['jpg', 'jpeg', 'png']
}

# Load credentials
try:
    aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
except Exception as e:
    logger.error(f"Failed to load AWS credentials: {e}")
    st.error("Configuration error. Please contact the administrator.")
    st.stop()

# Initialize S3 client
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {e}")
    st.error("Failed to connect to AWS services. Please try again later.")
    st.stop()


def apply_custom_css():
    """Apply custom CSS for professional UI styling"""
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #ffffff;
        }

        /* Header styling */
        .header-container {
            background-color: #1a1a1a;
            padding: 2rem 3rem;
            border-bottom: 3px solid #2c5aa0;
            margin-bottom: 2rem;
        }

        .header-title {
            color: #ffffff;
            font-size: 1.8rem;
            font-weight: 600;
            margin: 0;
            letter-spacing: 0.5px;
        }

        .header-subtitle {
            color: #b0b0b0;
            font-size: 0.95rem;
            margin-top: 0.5rem;
            font-weight: 400;
        }

        /* Status badges */
        .status-badge {
            display: inline-block;
            padding: 0.6rem 1.5rem;
            border-radius: 4px;
            font-weight: 600;
            font-size: 1rem;
            margin: 1.5rem 0;
            letter-spacing: 0.5px;
        }

        .status-pass {
            background-color: #0d7c3a;
            color: white;
            border-left: 4px solid #0a5c2b;
        }

        .status-fail {
            background-color: #c41e3a;
            color: white;
            border-left: 4px solid #8e1629;
        }

        /* Metric cards */
        .metric-card {
            background-color: #f8f9fa;
            padding: 1.2rem;
            border-radius: 4px;
            border-left: 3px solid #2c5aa0;
            margin: 0.5rem 0;
        }

        .metric-label {
            font-weight: 500;
            color: #666666;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 1.8rem;
            color: #1a1a1a;
            font-weight: 600;
        }

        /* Info card for welcome */
        .info-card {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
            margin: 1rem 0;
        }

        /* Progress message */
        .progress-message {
            background-color: #f0f4f8;
            padding: 1rem 1.5rem;
            border-radius: 4px;
            border-left: 3px solid #2c5aa0;
            margin: 1rem 0;
            color: #1a1a1a;
        }

        /* Section headers */
        .section-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1a1a1a;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e0e0e0;
        }
    </style>
    """, unsafe_allow_html=True)


def validate_image(image_data: bytes) -> Tuple[bool, str]:
    """Validate uploaded image"""
    try:
        # Check file size
        size_mb = len(image_data) / (1024 * 1024)
        if size_mb > CONFIG['MAX_FILE_SIZE_MB']:
            return False, f"File size exceeds {CONFIG['MAX_FILE_SIZE_MB']}MB limit"

        # Try to open image
        image = Image.open(BytesIO(image_data))

        # Verify it's a valid image
        image.verify()

        return True, "Valid image"
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False, f"Invalid image file: {str(e)}"


def upload_to_s3(image_data: bytes, image_key: str) -> Tuple[bool, str]:
    """Upload image to S3 with retry logic"""
    for attempt in range(CONFIG['MAX_RETRIES']):
        try:
            # Generate presigned POST URL
            response = s3_client.generate_presigned_post(
                Bucket=CONFIG['BUCKET_NAME'],
                Key=image_key,
                ExpiresIn=CONFIG['PRESIGNED_URL_EXPIRY']
            )

            # Upload file
            files = {'file': image_data}
            r = requests.post(
                response['url'],
                data=response['fields'],
                files=files,
                timeout=30
            )

            if r.status_code == 204:
                logger.info(f"Successfully uploaded image: {image_key}")
                return True, "Upload successful"
            else:
                logger.warning(f"Upload failed with status {r.status_code}, attempt {attempt + 1}")

        except Exception as e:
            logger.error(f"Upload error on attempt {attempt + 1}: {e}")
            if attempt < CONFIG['MAX_RETRIES'] - 1:
                time.sleep(1)  # Wait before retry

    return False, "Failed to upload image after multiple attempts"


def poll_for_result(labelled_image_key: str) -> Tuple[Optional[Image.Image], str]:
    """Poll S3 for processed image with intelligent wait"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    for attempt in range(CONFIG['MAX_POLLING_ATTEMPTS']):
        try:
            # Update progress
            progress = (attempt + 1) / CONFIG['MAX_POLLING_ATTEMPTS']
            progress_bar.progress(progress)

            elapsed_time = (attempt + 1) * CONFIG['POLLING_INTERVAL']
            status_text.markdown(
                f"<div class='progress-message'>Processing image... ({elapsed_time}s elapsed)</div>",
                unsafe_allow_html=True
            )

            # Try to get the processed image
            content_object = s3_client.get_object(
                Bucket=CONFIG['BUCKET_NAME'],
                Key=labelled_image_key
            )

            file_content = content_object['Body'].read()
            image = Image.open(BytesIO(file_content))

            progress_bar.progress(1.0)
            status_text.empty()
            logger.info(f"Successfully retrieved processed image: {labelled_image_key}")

            return image, "Success"

        except s3_client.exceptions.NoSuchKey:
            # Image not ready yet, continue polling
            time.sleep(CONFIG['POLLING_INTERVAL'])

        except Exception as e:
            logger.error(f"Error polling for result: {e}")
            time.sleep(CONFIG['POLLING_INTERVAL'])

    progress_bar.empty()
    status_text.empty()
    return None, "Timeout waiting for results"


def analyze_detection_result(image: Image.Image, image_key: str) -> Dict:
    """
    Analyze detection results from processed image
    In a real scenario, this might parse metadata from S3 or an API response
    For now, we'll return mock data - you can enhance this based on your backend
    """
    # TODO: Integrate with your actual backend API to get detection metadata
    # This is a placeholder that you can replace with actual API calls

    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_key': image_key,
        'status': 'PASS',  # or 'FAIL' based on actual detection
        'confidence': 0.95,
        'screws_detected': 12,
        'expected_screws': 12,
        'white_screws': 8,
        'green_screws': 4,
        'missing_screws': 0,
        'defects_found': 0
    }

    return result


def process_image(image_data: bytes) -> Tuple[bool, Optional[Image.Image], Optional[Dict], str]:
    """Main image processing pipeline"""

    # Validate image
    is_valid, validation_msg = validate_image(image_data)
    if not is_valid:
        return False, None, None, validation_msg

    # Generate unique image key
    unique_id = str(uuid.uuid4())
    image_key = f"{CONFIG['UNLABELLED_PREFIX']}{unique_id}.jpg"
    labelled_image_key = f"{CONFIG['LABELLED_PREFIX']}{unique_id}.jpg"

    # Upload to S3
    upload_success, upload_msg = upload_to_s3(image_data, image_key)
    if not upload_success:
        return False, None, None, upload_msg

    # Poll for processed result
    with st.spinner('Detection in progress...'):
        processed_image, poll_msg = poll_for_result(labelled_image_key)

    if processed_image is None:
        return False, None, None, poll_msg

    # Analyze results
    detection_results = analyze_detection_result(processed_image, image_key)

    return True, processed_image, detection_results, "Processing complete"


def display_results(original_image: bytes, processed_image: Image.Image, results: Dict):
    """Display results in a clean, professional layout"""

    # Display status badge
    status = results['status']
    status_class = 'status-pass' if status == 'PASS' else 'status-fail'

    st.markdown(
        f"<div class='status-badge {status_class}'>INSPECTION RESULT: {status}</div>",
        unsafe_allow_html=True
    )

    # Display images side by side
    st.markdown("<div class='section-header'>Image Comparison</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        st.image(original_image, use_container_width=True)

    with col2:
        st.markdown("**Detection Results**")
        st.image(processed_image, use_container_width=True)

    # Display detection metrics
    st.markdown("<div class='section-header'>Detection Metrics</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>Screws Detected</div>
                <div class='metric-value'>{results['screws_detected']}/{results['expected_screws']}</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>Confidence</div>
                <div class='metric-value'>{results['confidence']:.1%}</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>White Screws</div>
                <div class='metric-value'>{results['white_screws']}</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>Green Screws</div>
                <div class='metric-value'>{results['green_screws']}</div>
            </div>""",
            unsafe_allow_html=True
        )

    # Display detailed information
    with st.expander("View Detailed Information"):
        st.write(f"**Timestamp:** {results['timestamp']}")
        st.write(f"**Image ID:** {results['image_key'].split('/')[-1]}")
        st.write(f"**Missing Screws:** {results['missing_screws']}")
        st.write(f"**Defects Found:** {results['defects_found']}")

    # Download button
    buf = BytesIO()
    processed_image.save(buf, format='JPEG')
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Result Image",
        data=byte_im,
        file_name=f"screw_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
        mime="image/jpeg"
    )


def main():
    """Main application entry point"""

    # Page configuration
    st.set_page_config(
        page_title="Automotive Quality Inspection",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom styling
    apply_custom_css()

    # Header
    st.markdown("""
        <div class='header-container'>
            <h1 class='header-title'>Automotive Quality Inspection System</h1>
            <p class='header-subtitle'>Automated Screw Detection for Manufacturing Quality Control</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("assets/logo.png", use_container_width=True)
        st.markdown("---")
        st.markdown("### Inspection Options")

        option = st.selectbox(
            "Select input method:",
            (None, "Camera Input", "File Upload"),
            index=0
        )

        st.markdown("---")

        # Information section
        with st.expander("About"):
            st.write("""
            This system uses computer vision to detect and verify screws on car bumper components.

            **Supported Screw Types:**
            - White screws
            - Green screws

            **Features:**
            - Real-time detection
            - Quality verification
            - Confidence scoring
            - Detailed reporting
            """)

        with st.expander("Settings"):
            st.write(f"**Max File Size:** {CONFIG['MAX_FILE_SIZE_MB']}MB")
            st.write(f"**Supported Formats:** {', '.join(CONFIG['ALLOWED_EXTENSIONS'])}")
            st.write(f"**Max Wait Time:** {CONFIG['MAX_POLLING_ATTEMPTS'] * CONFIG['POLLING_INTERVAL']}s")

    # Main content area
    if option is None:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div class='info-card' style='text-align: center; padding: 3rem;'>
                    <h2>Get Started</h2>
                    <p style='font-size: 1.05rem; color: #666;'>
                        Select an input method from the sidebar to begin quality inspection
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Display sample images
            st.markdown("<div class='section-header'>Sample Images</div>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                try:
                    st.image("test_image/good.jpg", caption="Example: Pass", use_container_width=True)
                except:
                    pass
            with col_b:
                try:
                    st.image("test_image/not_good.jpeg", caption="Example: Fail", use_container_width=True)
                except:
                    pass

    elif option == "Camera Input":
        st.markdown("<div class='section-header'>Camera Input</div>", unsafe_allow_html=True)

        with st.sidebar:
            picture = st.camera_input("Take a picture of the component")

        if picture:
            picture_bytes = picture.getvalue()

            # Show original in sidebar
            with st.sidebar:
                st.success("Image captured")

            # Process image
            success, processed_image, results, message = process_image(picture_bytes)

            if success:
                display_results(picture_bytes, processed_image, results)
            else:
                st.error(f"Error: {message}")
                logger.error(f"Processing failed: {message}")

    elif option == "File Upload":
        st.markdown("<div class='section-header'>File Upload</div>", unsafe_allow_html=True)

        with st.sidebar:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=CONFIG['ALLOWED_EXTENSIONS'],
                help=f"Max file size: {CONFIG['MAX_FILE_SIZE_MB']}MB"
            )

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()

            # Show original in sidebar
            with st.sidebar:
                st.success("Image uploaded")

            # Process image
            success, processed_image, results, message = process_image(image_bytes)

            if success:
                display_results(image_bytes, processed_image, results)
            else:
                st.error(f"Error: {message}")
                logger.error(f"Processing failed: {message}")


if __name__ == "__main__":
    main()
