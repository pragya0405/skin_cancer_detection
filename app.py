import os
import pickle
import sqlite3
import uuid
from functools import wraps

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
DATABASE_PATH = os.path.join(BASE_DIR, "user_data.db")
MODEL_PATH = os.path.join(BASE_DIR, "ResNet50_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.pkl")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "images")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load the model once when the service starts.
model = load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, "rb") as class_file:
    class_names = pickle.load(class_file)


def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    image = load_img(image_path, target_size=(150, 150))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0) / 255.0
    probabilities = model.predict(image_array, verbose=0)[0]
    if len(probabilities) != len(class_names):
        raise ValueError("The model output does not match the saved class names.")
    class_index = int(np.argmax(probabilities))
    return class_names[class_index], float(probabilities[class_index] * 100)


def create_user_table():
    with get_db_connection() as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS user "
            "(name TEXT, password TEXT, mobile TEXT, email TEXT)"
        )


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("home"))
        return view(*args, **kwargs)

    return wrapped_view


create_user_table()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/userlog", methods=["GET", "POST"])
def userlog():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        password = request.form.get("password", "")
        with get_db_connection() as connection:
            user = connection.execute(
                "SELECT rowid, password FROM user WHERE name = ? ORDER BY rowid LIMIT 1",
                (name,),
            ).fetchone()
            if user is None:
                return render_template(
                    "index.html", msg="Sorry, incorrect credentials provided. Try again."
                )

            saved_password = user["password"] or ""
            try:
                authenticated = check_password_hash(saved_password, password)
            except (ValueError, TypeError):
                authenticated = False

            # Older rows stored plain-text passwords. Upgrade the matching row on login.
            if not authenticated and saved_password == password:
                authenticated = True
                connection.execute(
                    "UPDATE user SET password = ? WHERE rowid = ?",
                    (generate_password_hash(password, method="pbkdf2:sha256"), user["rowid"]),
                )

            if not authenticated:
                return render_template(
                    "index.html", msg="Sorry, incorrect credentials provided. Try again."
                )
        session["user_id"] = user["rowid"]
        return render_template("userlog.html")
    return render_template("home.html")


@app.route("/userreg", methods=["GET", "POST"])
def userreg():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        password = request.form.get("password", "")
        mobile = request.form.get("phone", "").strip()
        email = request.form.get("email", "").strip()
        if not all((name, password, mobile, email)):
            return render_template("index.html", msg="Please complete all registration fields.")

        with get_db_connection() as connection:
            connection.execute(
                "INSERT INTO user (name, password, mobile, email) VALUES (?, ?, ?, ?)",
                (name, generate_password_hash(password, method="pbkdf2:sha256"), mobile, email),
            )
        return render_template("index.html", msg="Successfully registered")
    return render_template("index.html")


@app.route("/userlog.html", methods=["GET"])
@login_required
def indexBt():
    return render_template("userlog.html")


@app.route("/graph.html", methods=["GET", "POST"])
@login_required
def graph():
    images = [
        url_for("static", filename="accuracy_plot.png"),
        url_for("static", filename="loss_plot.png"),
    ]
    content = ["Accuracy Graph", "Loss Graph (Error Message)"]
    return render_template("graph.html", images=images, content=content)


@app.route("/image", methods=["GET", "POST"])
@login_required
def image():
    if request.method != "POST":
        return render_template("userlog.html")

    uploaded_file = request.files.get("filename")
    if uploaded_file is None or not uploaded_file.filename:
        return render_template("userlog.html", msg="Choose an image file to analyse."), 400
    if not allowed_file(uploaded_file.filename):
        return render_template("userlog.html", msg="Use a JPG, PNG, BMP, or WebP image."), 400

    safe_name = secure_filename(uploaded_file.filename)
    stored_name = f"{uuid.uuid4().hex}_{safe_name}"
    image_path = os.path.join(UPLOAD_DIR, stored_name)
    uploaded_file.save(image_path)

    image_data = cv2.imread(image_path)
    if image_data is None:
        os.remove(image_path)
        return render_template("userlog.html", msg="The selected file is not a readable image."), 400

    try:
        predicted_class, confidence = predict_image(image_path)
    except Exception:
        app.logger.exception("Image prediction failed")
        os.remove(image_path)
        return render_template(
            "userlog.html", msg="The image could not be analysed. Please try another image."
        ), 500

    gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image_data, 250, 254)
    _, threshold = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image_data, -1, sharpening_kernel)
    output_images = {
        "gray": ("gray.jpg", gray_image),
        "edges": ("edges.jpg", edges),
        "threshold": ("threshold.jpg", threshold),
        "sharpened": ("sharpened.jpg", sharpened),
    }
    output_urls = {}
    for kind, (filename, output_image) in output_images.items():
        output_name = f"{uuid.uuid4().hex}_{filename}"
        cv2.imwrite(os.path.join(UPLOAD_DIR, output_name), output_image)
        output_urls[kind] = url_for("static", filename=f"images/{output_name}")

    treatment = "Medical Treatment:"
    treatments = []
    if predicted_class.lower() == "benign":
        treatments = [
            "A clinician can decide whether monitoring or removal is appropriate.",
            "Do not use an image classification result as a diagnosis.",
        ]
    elif predicted_class.lower() == "malignant":
        treatments = [
            "Please seek prompt assessment from a qualified dermatologist.",
            "Treatment decisions require clinical examination and, where appropriate, biopsy.",
        ]

    image_url = url_for("static", filename=f"images/{stored_name}")
    return render_template(
        "results.html",
        status=predicted_class,
        status2=f"{confidence:.2f}%",
        Treatment=treatment,
        Treatment1=treatments,
        ImageDisplay=image_url,
        ImageDisplay1=output_urls["gray"],
        ImageDisplay2=output_urls["edges"],
        ImageDisplay3=output_urls["threshold"],
        ImageDisplay4=output_urls["sharpened"],
    )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template("userlog.html", msg="Image must be 10 MB or smaller."), 413


@app.route("/logout")
def logout():
    session.clear()
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
