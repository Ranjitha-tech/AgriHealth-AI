
import streamlit as st
import sqlite3
import json
import hashlib
import hmac
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="AgriHealth AI | Healthy Plants, Happy Farmers",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"
DB_PATH = DATA_DIR / "analysis_history.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# DATABASE
# --------------------------------------------------
def get_connection():
    return sqlite3.connect(DB_PATH, timeout=10)


def init_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # A separate table keeps earlier project history intact.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                top_predictions TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)


init_database()


# --------------------------------------------------
# PASSWORD SECURITY
# --------------------------------------------------
def hash_password(password):
    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, 310000
    )
    return salt.hex() + ":" + derived.hex()


def verify_password(password, stored_hash):
    try:
        salt_hex, hash_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, 310000
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


# --------------------------------------------------
# MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    import tensorflow as tf

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if isinstance(class_names, dict):
        class_names = list(class_names.values())

    return model, class_names


def predict_leaf(image):
    model, class_names = load_model()

    # Resize the leaf image to the model's expected input size.
    input_shape = model.input_shape
    height = input_shape[1] or 224
    width = input_shape[2] or 224

    image = image.convert("RGB").resize((width, height))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    predictions = np.asarray(model.predict(array, verbose=0)[0])
    predictions = predictions.flatten()

    if len(predictions) != len(class_names):
        raise ValueError(
            "The number of model outputs does not match class_names.json."
        )

    # Supports the softmax classifier used by the trained project.
    top_indices = np.argsort(predictions)[::-1][:3]
    best_index = int(top_indices[0])

    top_three = [
        {
            "class": class_names[int(i)],
            "confidence": float(predictions[int(i)]) * 100,
        }
        for i in top_indices
    ]

    return class_names[best_index], float(predictions[best_index]) * 100, top_three


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
defaults = {
    "page": "Home",
    "user_id": None,
    "user_name": None,
    "user_email": None,
    "last_result": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# VISUAL DESIGN
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --green: #176b3a;
    --green-dark: #0d432d;
    --green-light: #eaf5e9;
    --text: #183b2b;
    --muted: #6a8174;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f6faf5;
    color: var(--text);
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif;
    color: var(--green-dark);
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #104d32, #092f24);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] button {
    border-radius: 10px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(110deg, #19834a, #126039);
    color: white;
    border: 0;
    border-radius: 11px;
    min-height: 43px;
    font-weight: 700;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #0d482d;
    color: white;
    transform: translateY(-1px);
}

div[data-testid="stForm"] {
    background: white;
    border: 1px solid #e0ebe1;
    border-radius: 18px;
    padding: 22px;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #e1ece2;
}

.hero {
    position: relative;
    overflow: hidden;
    min-height: 385px;
    padding: 48px 42px;
    border-radius: 24px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    background:
      linear-gradient(90deg, rgba(7,45,28,.91), rgba(8,57,33,.65), rgba(8,57,33,.16)),
      url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=85");
    background-size: cover;
    background-position: center;
    color: white;
    box-shadow: 0 16px 40px rgba(16, 70, 40, .14);
}

.hero h1 {
    color: white;
    font-size: clamp(2.2rem, 5vw, 3.7rem);
    line-height: 1.08;
    max-width: 650px;
    margin-bottom: 14px;
}

.hero p {
    color: #f0f8ef;
    font-size: 1.08rem;
    line-height: 1.7;
    max-width: 590px;
}

.hero .eyebrow {
    text-transform: uppercase;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: 2px;
    color: #c5f0c6;
    margin-bottom: 12px;
}

.hero .highlight {
    color: #b9f2b7;
}

.section-label {
    color: #23824a;
    font-size: .78rem;
    text-transform: uppercase;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.feature-card {
    height: 100%;
    background: white;
    padding: 23px;
    border: 1px solid #e0ebe1;
    border-radius: 18px;
}

.feature-icon {
    font-size: 1.8rem;
    margin-bottom: 10px;
}

.feature-card h3 {
    font-size: 1.05rem;
    margin-bottom: 7px;
}

.feature-card p {
    color: #6a8174;
    line-height: 1.6;
    font-size: .92rem;
}

.auth-panel {
    background: white;
    border: 1px solid #e0ebe1;
    border-radius: 22px;
    padding: 28px;
    box-shadow: 0 12px 36px rgba(19, 73, 42, .07);
}

.mini-banner {
    padding: 24px;
    border-radius: 18px;
    color: white;
    background:
      linear-gradient(90deg, rgba(8,52,32,.9), rgba(8,52,32,.56)),
      url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80");
    background-position: center;
    background-size: cover;
}

.footer-note {
    text-align: center;
    color: #718779;
    padding: 22px 0 4px;
    font-size: .85rem;
}

div[data-testid="stFileUploader"] {
    background: #f4faf4;
    border: 1px dashed #9ec8a5;
    border-radius: 14px;
    padding: 10px;
}

hr {
    border-color: #e3ece3;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def go_to(page):
    st.session_state.page = page
    st.rerun()


def create_user(full_name, email, password):
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO users
                   (full_name, email, password_hash, created_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    full_name.strip(),
                    email.strip().lower(),
                    hash_password(password),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def authenticate(email, password):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, full_name, email, password_hash FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()

    if row and verify_password(password, row[3]):
        return {"id": row[0], "name": row[1], "email": row[2]}

    return None


def save_analysis(filename, predicted_class, confidence, top_predictions):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO user_analyses
               (user_id, timestamp, filename, predicted_class, confidence, top_predictions)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                st.session_state.user_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                filename,
                predicted_class,
                confidence,
                json.dumps(top_predictions),
            ),
        )


def get_history():
    with get_connection() as conn:
        return conn.execute(
            """SELECT timestamp, filename, predicted_class, confidence, top_predictions
               FROM user_analyses
               WHERE user_id = ?
               ORDER BY id DESC""",
            (st.session_state.user_id,),
        ).fetchall()


def pretty_class(name):
    return name.replace("___", " — ").replace("__", " — ").replace("_", " ")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🌿 AgriHealth AI")
    st.caption("Healthy Plants • Happy Farmers")
    st.divider()

    if st.session_state.user_id:
        st.markdown(f"### Welcome, {st.session_state.user_name}!")
        st.caption(st.session_state.user_email)
        st.divider()

        if st.button("🏡  Home", use_container_width=True):
            go_to("Dashboard")
        if st.button("🔬  Analyze a Leaf", use_container_width=True):
            go_to("Analyze")
        if st.button("📋  Analysis History", use_container_width=True):
            go_to("History")
        if st.button("👤  My Account", use_container_width=True):
            go_to("Account")

        st.divider()
        if st.button("↪  Logout", use_container_width=True):
            for key in ("user_id", "user_name", "user_email", "last_result"):
                st.session_state[key] = None
            st.session_state.page = "Home"
            st.rerun()
    else:
        st.markdown("### Welcome, Farmer!")
        st.write("Your companion for healthier crops and better harvests.")
        if st.button("🔐  Login", use_container_width=True):
            go_to("Login")
        if st.button("🌱  Create Account", use_container_width=True):
            go_to("Register")

    st.markdown("---")
    st.caption("Built with care for the people who grow our food. 💚")


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
def home_page():
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">🌱 Technology with a farmer's heart</div>
        <h1>Healthy Plants.<br><span class="highlight">Happy Farmers.</span></h1>
        <p>
          Behind every harvest is a farmer's patience, hard work and hope.
          AgriHealth AI helps identify possible plant diseases early,
          so farmers can make more informed decisions and protect their crops.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Every healthy leaf supports a farmer's dream.")
        st.write(
            "From preparing the soil to caring for each plant, farming takes "
            "dedication every day. This project brings machine learning closer "
            "to that effort through accessible plant-image analysis."
        )
        if st.button("Get Started  →", key="home_start"):
            go_to("Register" if not st.session_state.user_id else "Dashboard")

    with right:
        st.markdown("""
        <div class="mini-banner">
          <div class="section-label" style="color:#c8f0cb;">OUR PURPOSE</div>
          <h2 style="color:white;">Support the hands that feed us.</h2>
          <p style="color:#f0f8ef;line-height:1.7;">
            Detect potential crop problems earlier, learn from results,
            and give every farmer a simpler way to begin.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-label">What you can do</div>', unsafe_allow_html=True)
    st.markdown("## Made for plant care, designed around farmers")

    c1, c2, c3 = st.columns(3)

    features = [
        ("🌿", "Detect plant diseases",
         "Upload a leaf photo and view the model's predicted class and confidence."),
        ("🛡️", "Make informed decisions",
         "Use the result as an early clue, then confirm it with a local agricultural expert."),
        ("📈", "Keep track of analyses",
         "Sign in to review your previous predictions and compare your saved results."),
    ]

    for col, feature in zip((c1, c2, c3), features):
        with col:
            st.markdown(
                f"""<div class="feature-card">
                <div class="feature-icon">{feature[0]}</div>
                <h3>{feature[1]}</h3>
                <p>{feature[2]}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    st.info(
        "This is a machine-learning project, not a substitute for professional "
        "crop diagnosis. Predictions can be wrong; confirm treatment decisions "
        "with a qualified agricultural adviser."
    )
    st.markdown(
        '<div class="footer-note">🌾 Respect for farmers. Care for every crop. '
        'Hope for every harvest.</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------
def login_page():
    _, center, _ = st.columns([1, 1.15, 1])

    with center:
        st.markdown('<div class="auth-panel">', unsafe_allow_html=True)
        st.markdown("## 🌿 Welcome back!")
        st.write("Sign in to continue to your plant health dashboard.")

        with st.form("login_form"):
            email = st.text_input("Email address", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if not email.strip() or not password:
                st.error("Please enter your email and password.")
            else:
                user = authenticate(email, password)
                if user:
                    st.session_state.user_id = user["id"]
                    st.session_state.user_name = user["name"]
                    st.session_state.user_email = user["email"]
                    st.session_state.page = "Dashboard"
                    st.rerun()
                else:
                    st.error("Email or password is incorrect.")

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("New to AgriHealth AI?")
        if st.button("Create a new account", use_container_width=True):
            go_to("Register")


# --------------------------------------------------
# REGISTER PAGE
# --------------------------------------------------
def register_page():
    _, center, _ = st.columns([1, 1.2, 1])

    with center:
        st.markdown('<div class="auth-panel">', unsafe_allow_html=True)
        st.markdown("## Create your account")
        st.write("Join us in caring for healthier crops and happier harvests.")

        with st.form("register_form"):
            full_name = st.text_input("Full name", placeholder="Enter your name")
            email = st.text_input("Email address", placeholder="you@example.com")
            password = st.text_input(
                "Create password", type="password",
                help="Use at least 8 characters."
            )
            confirm = st.text_input("Confirm password", type="password")
            accepted = st.checkbox("I understand that predictions may be inaccurate.")
            submitted = st.form_submit_button("Create Account", use_container_width=True)

        if submitted:
            if not full_name.strip() or not email.strip() or not password:
                st.error("Please complete all fields.")
            elif "@" not in email or "." not in email.split("@")[-1]:
                st.error("Please enter a valid email address.")
            elif len(password) < 8:
                st.error("Your password must contain at least 8 characters.")
            elif password != confirm:
                st.error("The passwords do not match.")
            elif not accepted:
                st.error("Please confirm that you understand the prediction limitations.")
            elif not create_user(full_name, email, password):
                st.error("An account with this email already exists. Please log in.")
            else:
                st.success("Account created! You can now log in.")
                st.session_state.page = "Login"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("Already have an account?")
        if st.button("Go to Login", use_container_width=True):
            go_to("Login")


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
def dashboard_page():
    st.markdown(f"## Welcome, {st.session_state.user_name} 🌱")
    st.write("Every plant deserves care. Let's take a look at your crops.")

    st.markdown("""
    <div class="mini-banner">
      <h2 style="color:white;">Small steps. Healthier crops.</h2>
      <p style="color:#eff8ed;">Upload a clear leaf image to explore a possible
      disease prediction from your trained AI model.</p>
    </div>
    """, unsafe_allow_html=True)

    history = get_history()
    total = len(history)
    healthy = sum(
        1 for row in history
        if "healthy" in row[2].lower()
    )

    st.write("")
    a, b, c = st.columns(3)
    a.metric("Your analyses", total)
    b.metric("Healthy predictions", healthy)
    c.metric("Model validation accuracy", "91.23%")

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
          <div class="feature-icon">🔬</div>
          <h3>Analyze a leaf</h3>
          <p>Upload a leaf photo to receive a prediction and confidence score.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Analyze a Leaf  →", use_container_width=True):
            go_to("Analyze")

    with col2:
        st.markdown("""
        <div class="feature-card">
          <div class="feature-icon">📋</div>
          <h3>Your analysis history</h3>
          <p>Review saved predictions from your previous sessions.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View History  →", use_container_width=True):
            go_to("History")


# --------------------------------------------------
# ANALYZE A LEAF
# --------------------------------------------------
def analyze_page():
    st.markdown("## 🔬 Analyze a leaf")
    st.write(
        "Choose a clear photo of a single leaf. Good lighting and a focused "
        "image can help the model produce a more useful prediction."
    )

    uploaded = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG.",
    )

    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
        except Exception:
            st.error("This image could not be opened. Please choose another image.")
            return

        left, right = st.columns([1, 1.15])

        with left:
            st.image(image, caption="Your selected leaf image", use_container_width=True)

        with right:
            st.markdown("### Ready to check your leaf?")
            st.write("Your image is processed for prediction. The photo itself is not saved by this app.")
            if st.button("🌿 Analyze Image", use_container_width=True):
                with st.spinner("Examining the leaf image..."):
                    try:
                        predicted_class, confidence, top_three = predict_leaf(image)
                        result = {
                            "filename": uploaded.name,
                            "predicted_class": predicted_class,
                            "confidence": confidence,
                            "top_predictions": top_three,
                        }
                        save_analysis(
                            uploaded.name, predicted_class, confidence, top_three
                        )
                        st.session_state.last_result = result
                        st.success("Analysis completed and saved to your history.")
                    except Exception as error:
                        st.error(f"Could not analyze the image: {error}")

    result = st.session_state.last_result

    if result:
        st.divider()
        st.markdown("## Your prediction")
        is_healthy = "healthy" in result["predicted_class"].lower()

        if is_healthy:
            st.success("The model predicts a healthy leaf.")
        else:
            st.warning(
                "The model detected a possible disease. Please confirm the result "
                "before choosing a treatment."
            )

        st.markdown(f"### {pretty_class(result['predicted_class'])}")
        st.metric("Model confidence", f"{result['confidence']:.2f}%")
        st.caption(
            "Confidence is the model's score for its prediction, not a guarantee "
            "that the diagnosis is correct."
        )

        st.markdown("### Top 3 predictions")
        for item in result["top_predictions"]:
            st.write(
                f"**{pretty_class(item['class'])}** — "
                f"{item['confidence']:.2f}%"
            )
            st.progress(
                min(max(item["confidence"] / 100, 0.0), 1.0)
            )

        st.info(
            "Next step: inspect the plant and nearby leaves, compare the symptoms, "
            "and consult a local agricultural officer or qualified expert before "
            "applying pesticides. This app does not prescribe a chemical treatment."
        )

        if st.button("Analyze another image", use_container_width=True):
            st.session_state.last_result = None
            st.rerun()


# --------------------------------------------------
# ANALYSIS HISTORY
# --------------------------------------------------
def history_page():
    st.markdown("## 📋 Your analysis history")
    st.write("Your saved predictions are listed below, newest first.")

    history = get_history()

    if not history:
        st.info("No analyses yet. Upload your first leaf image to get started.")
        if st.button("Analyze your first leaf"):
            go_to("Analyze")
        return

    st.metric("Total saved analyses", len(history))

    export_rows = []
    for timestamp, filename, predicted, confidence, top_json in history:
        export_rows.append({
            "Date and time": timestamp,
            "Filename": filename,
            "Prediction": pretty_class(predicted),
            "Confidence (%)": round(confidence, 2),
            "Top predictions": top_json,
        })

    import pandas as pd

    dataframe = pd.DataFrame(export_rows)
    st.dataframe(dataframe, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Export history as CSV",
        dataframe.drop(columns=["Top predictions"]).to_csv(index=False).encode("utf-8"),
        file_name="agrihealth_analysis_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("View individual prediction details"):
        for timestamp, filename, predicted, confidence, top_json in history:
            st.markdown(f"**{pretty_class(predicted)}**")
            st.caption(f"{timestamp} · {filename} · {confidence:.2f}% confidence")
            try:
                for item in json.loads(top_json):
                    st.write(
                        f"- {pretty_class(item['class'])}: "
                        f"{item['confidence']:.2f}%"
                    )
            except (ValueError, TypeError):
                st.caption("Detailed prediction scores are unavailable.")
            st.divider()


# --------------------------------------------------
# ACCOUNT PAGE
# --------------------------------------------------
def account_page():
    st.markdown("## 👤 My account")
    st.write("Your AgriHealth AI account details.")

    st.markdown(f"**Name:** {st.session_state.user_name}")
    st.markdown(f"**Email:** {st.session_state.user_email}")

    st.info(
        "Your account and analysis history are stored in a local SQLite database. "
        "This version is intended for a local project demonstration. It does not "
        "provide email verification, password recovery, or production-grade "
        "multi-user hosting security."
    )

    if st.button("Return to Dashboard"):
        go_to("Dashboard")


# --------------------------------------------------
# ROUTING
# --------------------------------------------------
page = st.session_state.page

if page == "Home":
    home_page()
elif page == "Login":
    login_page()
elif page == "Register":
    register_page()
elif page == "Dashboard":
    if st.session_state.user_id:
        dashboard_page()
    else:
        go_to("Login")
elif page == "Analyze":
    if st.session_state.user_id:
        analyze_page()
    else:
        go_to("Login")
elif page == "History":
    if st.session_state.user_id:
        history_page()
    else:
        go_to("Login")
elif page == "Account":
    if st.session_state.user_id:
        account_page()
    else:
        go_to("Login")
else:
    home_page()