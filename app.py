
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import textwrap
from PIL import Image

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. HTML RENDERING HELPER
# ==========================================

def render_html(content):
    st.markdown(
        textwrap.dedent(content).strip(),
        unsafe_allow_html=True
    )

# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "plant_disease_model.keras"
    )

    with open("class_names.json", "r") as file:
        class_names = json.load(file)

    return model, class_names


try:
    model, class_names = load_model()

except Exception as e:

    st.error("Unable to load the trained model.")

    st.write("Check that these files exist:")

    st.code("""
plant_disease_model.keras
class_names.json
""")

    st.stop()


# ==========================================
# 4. DISEASE INFORMATION
# ==========================================

disease_info = {

    "Tomato___Early_blight": {
        "symptoms": "Brown circular spots with concentric rings, often surrounded by yellow areas.",
        "cause": "Fungal infection caused by Alternaria solani.",
        "prevention": "Avoid overhead watering, improve airflow and remove infected leaves.",
        "care": "Remove badly affected leaves. Keep foliage dry and seek local agricultural advice."
    },

    "Tomato___Late_blight": {
        "symptoms": "Water-soaked patches that turn brown or black. White growth may appear in humid conditions.",
        "cause": "Caused by Phytophthora infestans.",
        "prevention": "Avoid prolonged leaf wetness and use healthy planting material.",
        "care": "Remove infected plant material and seek agricultural advice promptly."
    },

    "Tomato___Bacterial_spot": {
        "symptoms": "Small dark spots on leaves and sometimes raised spots on fruit.",
        "cause": "Bacterial infection caused by Xanthomonas species.",
        "prevention": "Use disease-free seeds and sanitize gardening tools.",
        "care": "Remove severely affected plant material and avoid spreading debris."
    },

    "Tomato___Septoria_leaf_spot": {
        "symptoms": "Small circular spots with grey or tan centres and dark edges.",
        "cause": "Fungal infection caused by Septoria lycopersici.",
        "prevention": "Remove infected leaves, avoid overhead watering and maintain plant spacing.",
        "care": "Remove affected foliage and keep leaves dry."
    },

    "Tomato___Leaf_Mold": {
        "symptoms": "Yellow patches on upper leaf surfaces and olive-green growth underneath.",
        "cause": "Fungal disease associated with high humidity.",
        "prevention": "Improve ventilation and reduce humidity.",
        "care": "Remove badly affected leaves and improve airflow."
    },

    "Potato___Early_blight": {
        "symptoms": "Dark brown leaf spots with concentric rings.",
        "cause": "Fungal infection caused by Alternaria solani.",
        "prevention": "Rotate crops and avoid water stress.",
        "care": "Remove infected foliage and seek local crop advice."
    },

    "Potato___Late_blight": {
        "symptoms": "Dark water-soaked leaf patches that can spread rapidly.",
        "cause": "Caused by Phytophthora infestans.",
        "prevention": "Use healthy seed potatoes and avoid prolonged leaf wetness.",
        "care": "Act quickly to limit spread and consult an agricultural expert."
    },

    "Apple___Apple_scab": {
        "symptoms": "Dark olive or black spots on leaves and scabby lesions on fruit.",
        "cause": "Fungal infection caused by Venturia inaequalis.",
        "prevention": "Remove fallen infected leaves and improve airflow.",
        "care": "Remove infected fallen material and seek orchard management advice."
    },

    "Apple___Black_rot": {
        "symptoms": "Dark spots on leaves and brown or black rotting areas on fruit.",
        "cause": "Fungal infection caused by Botryosphaeria obtusa.",
        "prevention": "Remove dead branches and infected plant material.",
        "care": "Prune affected branches using clean tools."
    },

    "Corn___Common_rust": {
        "symptoms": "Small reddish-brown raised pustules on corn leaves.",
        "cause": "Fungal infection caused by Puccinia sorghi.",
        "prevention": "Grow resistant varieties and monitor plants regularly.",
        "care": "Seek local crop advice if symptoms spread."
    },

    "Grape___Black_rot": {
        "symptoms": "Brown leaf spots and shrivelled, darkened berries.",
        "cause": "Fungal infection caused by Guignardia bidwellii.",
        "prevention": "Remove infected berries and improve airflow.",
        "care": "Remove infected plant material and maintain vineyard sanitation."
    },

    "Pepper,_bell___Bacterial_spot": {
        "symptoms": "Small dark spots on leaves and raised lesions on fruit.",
        "cause": "Bacterial infection caused by Xanthomonas species.",
        "prevention": "Use disease-free seeds and avoid handling wet plants.",
        "care": "Remove infected plant material and sanitize tools."
    },

    "Squash___Powdery_mildew": {
        "symptoms": "White powder-like patches on leaf surfaces.",
        "cause": "Fungal infection caused by powdery mildew fungi.",
        "prevention": "Provide good spacing, airflow and sunlight.",
        "care": "Remove severely affected leaves and seek suitable control advice."
    },

    "Strawberry___Leaf_scorch": {
        "symptoms": "Purple or reddish spots that may develop brown centres.",
        "cause": "Fungal infection caused by Diplocarpon earlianum.",
        "prevention": "Remove infected leaves and maintain good plant spacing.",
        "care": "Remove affected foliage and avoid prolonged leaf wetness."
    }
}


# ==========================================
# 5. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main background */

.stApp {
    background: #0b1712;
    color: #edf5ef;
}

/* Top header background - fixes white space */

header[data-testid="stHeader"] {
    background: #0b1712 !important;
}

[data-testid="stDecoration"] {
    display: none;
}

/* Main content spacing */

.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #101f18;
    border-right: 1px solid #294332;
}

section[data-testid="stSidebar"] * {
    color: #e5eee7;
}

/* Headings */

h1, h2, h3 {
    color: #e9f5eb !important;
    letter-spacing: -0.5px;
}

p, label {
    color: #b9cbbd;
}

/* Brand */

.brand {
    font-size: 40px;
    font-weight: 800;
    color: #b7e4c1;
    letter-spacing: -1px;
}

.brand-sub {
    color: #8ca797;
    font-size: 13px;
    margin-top: -8px;
    letter-spacing: 1px;
}

/* Hero section */

.hero {
    position: relative;
    overflow: hidden;

    background:
        radial-gradient(
            circle at 90% 20%,
            #477a503d,
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #173b29,
            #10271c
        );

    border: 1px solid #426c4d;
    border-radius: 22px;
    padding: 38px;
    margin: 12px 0 28px 0;

    box-shadow: 0 10px 35px #00000025;
}

.hero h1 {
    font-size: 36px;
    margin-bottom: 12px;
}

.hero p {
    color: #c3d9c9;
    font-size: 16px;
    max-width: 750px;
    line-height: 1.7;
}

/* Dashboard cards */

.info-card {
    background: linear-gradient(
        145deg,
        #192e21,
        #12231a
    );

    border: 1px solid #2d4936;
    padding: 24px;
    border-radius: 18px;
    min-height: 135px;

    transition: all 0.25s ease;

    box-shadow: 0 5px 18px #00000018;
}

.info-card:hover {
    transform: translateY(-5px);
    border-color: #79b989;
    box-shadow: 0 10px 25px #00000035;
}

.info-card h3 {
    color: #a9dcb5 !important;
    font-size: 17px;
}

.info-card p {
    color: #b9cbbd;
    font-size: 14px;
    line-height: 1.6;
}

/* Prediction card */

.result-card {
    background: linear-gradient(
        135deg,
        #173b29,
        #1c3526
    );

    border: 1px solid #60966b;
    border-radius: 20px;
    padding: 26px;
    margin: 15px 0;

    box-shadow: 0 8px 25px #00000025;
}

.result-card h2 {
    color: #b8e8c3 !important;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background: #14251b;
    border: 1px dashed #78b987;
    border-radius: 18px;
    padding: 20px;
    transition: 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #b8e8c3;
    background: #192f22;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        135deg,
        #91c99d,
        #65a978
    );

    color: #10251a !important;
    border: none;
    border-radius: 12px;
    padding: 13px 22px;
    font-weight: 700;

    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #b5e5bf;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px #78b98735;
}

/* Select box */

div[data-baseweb="select"] > div {
    background: #182c20;
    border-color: #385b41;
}

/* Metrics */

[data-testid="stMetric"] {
    background: #14251b;
    border: 1px solid #2b4433;
    padding: 18px;
    border-radius: 16px;
}

[data-testid="stMetricValue"] {
    color: #b8e8c3;
}

/* Progress bars */

.stProgress > div > div > div > div {
    background: #83c391;
}

/* Uploaded image */

[data-testid="stImage"] img {
    border-radius: 16px;
}

/* Divider */

hr {
    border-color: #294132;
}

/* Footer */

.footer {
    text-align: center;
    color: #819889;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 6. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    render_html("""
        <div class="brand">🌿 PlantCare</div>
        <div class="brand-sub">
            INTELLIGENT PLANT HEALTH SYSTEM
        </div>
    """)

    st.write("")
    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Home",
            "Disease Detection",
            "Disease Guide",
            "About"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### AI MODEL")

    render_html("""
        <div class="info-card">
            <h3>MobileNetV2</h3>
            <p>Deep learning model</p>
            <p>39 classification categories</p>
        </div>
    """)

    st.write("")
    st.caption("PlantCare AI • 2026")


# ==========================================
# 7. MAIN HEADER
# ==========================================

render_html("""
    <div style="margin-bottom:25px">
        <div class="brand">PlantCare AI 🌿</div>
        <div class="brand-sub">
            SMARTER PLANT HEALTH. BETTER GROWTH.
        </div>
    </div>
""")


# ==========================================
# 8. HOME PAGE
# ==========================================

if page == "Home":

    render_html("""
        <div class="hero">
            <h1>Give Your Plants a Healthier Future 🌱</h1>
            <p>
                Detect plant diseases using artificial intelligence.
                Upload a leaf image, analyze its condition and explore
                useful plant-care information.
            </p>
        </div>
    """)

    st.subheader("Your Plant Health Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:

        render_html("""
            <div class="info-card">
                <h3>🧠 AI Model</h3>
                <h2>MobileNetV2</h2>
                <p>Transfer learning architecture</p>
            </div>
        """)

    with col2:

        render_html("""
            <div class="info-card">
                <h3>🌿 Categories</h3>
                <h2>39</h2>
                <p>Plant and disease categories</p>
            </div>
        """)

    with col3:

        render_html("""
            <div class="info-card">
                <h3>📊 Validation Accuracy</h3>
                <h2>96.27%</h2>
                <p>Measured on validation dataset</p>
            </div>
        """)

    st.write("")
    st.subheader("Explore PlantCare AI")

    col1, col2 = st.columns(2)

    with col1:

        render_html("""
            <div class="info-card">
                <h3>🔍 Disease Detection</h3>
                <p>
                    Upload a leaf image and let our AI model
                    predict its likely disease category.
                </p>
            </div>
        """)

    with col2:

        render_html("""
            <div class="info-card">
                <h3>📚 Disease Guide</h3>
                <p>
                    Learn about symptoms, causes,
                    prevention and plant care.
                </p>
            </div>
        """)


# ==========================================
# 9. DISEASE DETECTION PAGE
# ==========================================

elif page == "Disease Detection":

    render_html("""
        <div class="hero">
            <h1>Plant Disease Detection 🔬</h1>
            <p>
                Upload a clear image of a plant leaf
                to analyze its likely health condition.
            </p>
        </div>
    """)

    st.markdown("### 📸 Upload your leaf image")

    st.caption("Supported formats: JPG, JPEG and PNG")

    uploaded_file = st.file_uploader(
        "Choose a plant image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

        except Exception:
            st.error("Unable to read this image. Please upload another.")
            st.stop()

        col1, col2 = st.columns([1, 1])

        with col1:

            st.subheader("Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("AI Analysis")

            render_html("""
                <div class="info-card">
                    <h3>Image uploaded successfully</h3>
                    <p>
                        Your image is ready for AI analysis.
                        Click the button below to detect the condition.
                    </p>
                </div>
            """)

            st.write("")

            detect_button = st.button(
                "🔍 Analyze Plant Image",
                type="primary"
            )

        if detect_button:

            with st.spinner("AI is analyzing your image..."):

                image_resized = image.resize((224, 224))

                image_array = np.array(image_resized)

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                predictions = model.predict(
                    image_array,
                    verbose=0
                )

            predicted_index = int(
                np.argmax(predictions[0])
            )

            predicted_class = class_names[predicted_index]

            confidence = float(
                predictions[0][predicted_index]
            ) * 100

            st.divider()

            st.subheader("Analysis Results")

            if predicted_class == "Background_without_leaves":

                st.warning(
                    "The image may not contain a recognizable plant leaf. "
                    "Please upload a clear leaf image."
                )

            else:

                parts = predicted_class.split("___")

                plant_name = parts[0].replace("_", " ")

                condition = parts[1].replace("_", " ")

                render_html(f"""
                    <div class="result-card">
                        <h2>🌿 {plant_name}</h2>
                        <h3>Prediction: {condition}</h3>
                        <h3>Confidence: {confidence:.2f}%</h3>
                    </div>
                """)

                if predicted_class in disease_info:

                    st.subheader("Recommended Care")

                    st.info(
                        disease_info[predicted_class]["care"]
                    )

                else:

                    st.info(
                        "Check the Disease Guide for more "
                        "information about this category."
                    )

            st.subheader("Top 3 Predictions")

            top_indices = np.argsort(
                predictions[0]
            )[-3:][::-1]

            for index in top_indices:

                label = class_names[index]

                score = float(
                    predictions[0][index]
                ) * 100

                st.write(
                    label.replace(
                        "___", " - "
                    ).replace("_", " ")
                )

                st.progress(
                    min(score / 100, 1.0)
                )

                st.caption(f"{score:.2f}%")

            st.caption(
                "Predictions are educational screening results, "
                "not a confirmed plant-health diagnosis."
            )


# ==========================================
# 10. DISEASE GUIDE PAGE
# ==========================================

elif page == "Disease Guide":

    render_html("""
        <div class="hero">
            <h1>Plant Disease Library 📚</h1>
            <p>
                Explore plant diseases, their symptoms,
                causes and prevention methods.
            </p>
        </div>
    """)

    disease_options = {
        key: key.replace(
            "___", " - "
        ).replace("_", " ")
        for key in disease_info
    }

    selected_disease = st.selectbox(
        "Choose a disease",
        list(disease_options.keys()),
        format_func=lambda x: disease_options[x]
    )

    info = disease_info[selected_disease]

    st.divider()

    st.subheader(
        disease_options[selected_disease]
    )

    col1, col2 = st.columns(2)

    with col1:

        render_html(f"""
            <div class="info-card">
                <h3>🔍 Symptoms</h3>
                <p>{info["symptoms"]}</p>
            </div>
        """)

    with col2:

        render_html(f"""
            <div class="info-card">
                <h3>🦠 Cause</h3>
                <p>{info["cause"]}</p>
            </div>
        """)

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        render_html(f"""
            <div class="info-card">
                <h3>🛡️ Prevention</h3>
                <p>{info["prevention"]}</p>
            </div>
        """)

    with col2:

        render_html(f"""
            <div class="info-card">
                <h3>💡 Recommended Care</h3>
                <p>{info["care"]}</p>
            </div>
        """)

    st.write("")

    st.caption(
        "This guide covers selected diseases. "
        "Consult an agricultural expert before applying treatments."
    )


# ==========================================
# 11. ABOUT PAGE
# ==========================================

elif page == "About":

    render_html("""
        <div class="hero">
            <h1>About PlantCare AI 🌿</h1>
            <p>
                An AI-powered plant disease screening
                system built using deep learning.
            </p>
        </div>
    """)

    st.subheader("Project Overview")

    st.write("""
        PlantCare AI is designed to identify likely plant diseases
        from leaf images using a deep learning model.

        The system uses MobileNetV2 transfer learning and
        the PlantVillage dataset.
    """)

    st.subheader("Technology Stack")

    technologies = [
        "Python",
        "TensorFlow",
        "Keras",
        "MobileNetV2",
        "Streamlit",
        "NumPy",
        "Pillow"
    ]

    cols = st.columns(4)

    for i, tech in enumerate(technologies):

        with cols[i % 4]:

            render_html(f"""
                <div class="info-card" style="min-height:75px">
                    <h3>{tech}</h3>
                </div>
            """)

    st.write("")

    st.subheader("Model Information")

    st.write("Architecture: MobileNetV2")
    st.write("Dataset: PlantVillage")
    st.write("Number of classes: 39")
    st.write("Validation accuracy: 96.27%")

    st.info(
        "Validation performance on PlantVillage images "
        "does not guarantee the same accuracy on real-world field images."
    )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

render_html("""
    <div class="footer">
        🌿 PlantCare AI<br>
        Intelligent Plant Disease Detection System<br>
        Built with Python, TensorFlow and Streamlit
    </div>
""")
