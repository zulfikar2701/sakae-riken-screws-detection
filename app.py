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
    """Apply custom CSS for modern UI styling"""
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #f8f9fa;
        }

        /* Header styling */
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .header-title {
            color: white;
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0;
            text-align: center;
        }

        .header-subtitle {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1rem;
            text-align: center;
            margin-top: 0.5rem;
        }

        /* Status badges */
        .status-badge {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.2rem;
            text-align: center;
            margin: 1rem 0;
        }

        .status-pass {
            background-color: #28a745;
            color: white;
        }

        .status-fail {
            background-color: #dc3545;
            color: white;
        }

        .status-processing {
            background-color: #ffc107;
            color: #333;
        }

        /* Info cards */
        .info-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        }

        .info-label {
            font-weight: bold;
            color: #667eea;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .info-value {
            font-size: 1.5rem;
            color: #333;
            margin-top: 0.5rem;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        /* Image container */
        .image-container {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        }

        /* Progress message */
        .progress-message {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
            margin: 1rem 0;
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
                f"<div class='progress-message'>⏳ Processing image... ({elapsed_time}s elapsed)</div>",
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
    with st.spinner('🔍 AI Detection in Progress...'):
        processed_image, poll_msg = poll_for_result(labelled_image_key)

    if processed_image is None:
        return False, None, None, poll_msg

    # Analyze results
    detection_results = analyze_detection_result(processed_image, image_key)

    return True, processed_image, detection_results, "Processing complete"


def display_results(original_image: bytes, processed_image: Image.Image, results: Dict):
    """Display results in a modern, informative layout"""

    # Display status badge
    status = results['status']
    status_class = 'status-pass' if status == 'PASS' else 'status-fail'
    status_icon = '✅' if status == 'PASS' else '❌'

    st.markdown(
        f"<div class='status-badge {status_class}'>{status_icon} Quality Check: {status}</div>",
        unsafe_allow_html=True
    )

    # Display images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📸 Original Image")
        st.image(original_image, use_container_width=True)

    with col2:
        st.markdown("### 🎯 Detection Results")
        st.image(processed_image, use_container_width=True)

    # Display detection metrics
    st.markdown("### 📊 Detection Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""<div class='info-card'>
                <div class='info-label'>Screws Detected</div>
                <div class='info-value'>{results['screws_detected']}/{results['expected_screws']}</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""<div class='info-card'>
                <div class='info-label'>Confidence</div>
                <div class='info-value'>{results['confidence']:.1%}</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""<div class='info-card'>
                <div class='info-label'>White Screws</div>
                <div class='info-value'>{results['white_screws']}</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""<div class='info-card'>
                <div class='info-label'>Green Screws</div>
                <div class='info-value'>{results['green_screws']}</div>
            </div>""",
            unsafe_allow_html=True
        )

    # Display detailed information
    with st.expander("📋 Detailed Information"):
        st.write(f"**Timestamp:** {results['timestamp']}")
        st.write(f"**Image ID:** {results['image_key'].split('/')[-1]}")
        st.write(f"**Missing Screws:** {results['missing_screws']}")
        st.write(f"**Defects Found:** {results['defects_found']}")

    # Download button
    buf = BytesIO()
    processed_image.save(buf, format='JPEG')
    byte_im = buf.getvalue()

    st.download_button(
        label="⬇️ Download Result Image",
        data=byte_im,
        file_name=f"screw_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
        mime="image/jpeg"
    )


def main():
    """Main application entry point"""

    # Page configuration
    st.set_page_config(
        page_title="Automotive Quality Inspection",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom styling
    apply_custom_css()

    # Header
    st.markdown("""
        <div class='header-container'>
            <h1 class='header-title'>🔧 Automotive Quality Inspection System</h1>
            <p class='header-subtitle'>AI-Powered Screw Detection for Car Bumper Components</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("assets/logo.png", use_container_width=True)
        st.markdown("---")
        st.markdown("### 🎯 Inspection Options")

        option = st.selectbox(
            "Select input method:",
            (None, "📷 Take a Picture", "📁 Upload a Picture"),
            index=0
        )

        st.markdown("---")

        # Information section
        with st.expander("ℹ️ About"):
            st.write("""
            This system uses advanced AI to detect and verify screws on car bumper components.

            **Supported Screw Types:**
            - White screws
            - Green screws

            **Features:**
            - Real-time detection
            - Quality verification
            - Confidence scoring
            - Detailed reporting
            """)

        with st.expander("⚙️ Settings"):
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
                    <h2>👈 Get Started</h2>
                    <p style='font-size: 1.1rem; color: #666;'>
                        Select an input method from the sidebar to begin quality inspection
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Display sample images
            st.markdown("### 📚 Sample Images")
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

    elif option == "📷 Take a Picture":
        st.markdown("### 📷 Camera Input")

        with st.sidebar:
            picture = st.camera_input("Take a picture of the component")

        if picture:
            picture_bytes = picture.getvalue()

            # Show original in sidebar
            with st.sidebar:
                st.success("✅ Image captured!")

            # Process image
            success, processed_image, results, message = process_image(picture_bytes)

            if success:
                display_results(picture_bytes, processed_image, results)
            else:
                st.error(f"❌ {message}")
                logger.error(f"Processing failed: {message}")

    elif option == "📁 Upload a Picture":
        st.markdown("### 📁 File Upload")

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
                st.success("✅ Image uploaded!")

            # Process image
            success, processed_image, results, message = process_image(image_bytes)

            if success:
                display_results(image_bytes, processed_image, results)
            else:
                st.error(f"❌ {message}")
                logger.error(f"Processing failed: {message}")


if __name__ == "__main__":
    main()
