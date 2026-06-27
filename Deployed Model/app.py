import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import send_file
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.platypus import Image
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

from datetime import datetime

# Import AI Assistant
from ai_assistant import AI_ASSISTANT


# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)


# -----------------------------
# Load Trained Model
# -----------------------------


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "dr_multiclass_model.h5"
)

model = load_model(MODEL_PATH)


CLASS_NAMES = [
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "No Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy",
    "Severe Diabetic Retinopathy"
]
last_prediction = {}

# -----------------------------
# Prediction Function
# -----------------------------
def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    from tensorflow.keras.applications.efficientnet import preprocess_input
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)[0]

    class_index = int(np.argmax(preds))

    confidence = round(float(preds[class_index]) * 100, 2)

    stage = CLASS_NAMES[class_index]

    probabilities = {}

    for i, name in enumerate(CLASS_NAMES):
        probabilities[name] = round(float(preds[i]) * 100, 2)

    return stage, confidence, probabilities


# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return {"error": "No file uploaded"}

    f = request.files['file']

    if f.filename == '':
        return {"error": "No selected file"}

    basepath = os.path.dirname(__file__)
    upload_dir = os.path.join(basepath, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, secure_filename(f.filename))
    f.save(file_path)

    # Model prediction
    stage, confidence, probabilities = model_predict(file_path)

    # AI Assistant response
    assistant_data = AI_ASSISTANT[stage]
    global last_prediction

    last_prediction = {
    "stage": stage,
    "confidence": confidence,
    "info": assistant_data["info"],
    "precautions": assistant_data["precautions"],
    "diet": assistant_data["diet"],
    "advice": assistant_data["advice"],
    "probabilities": probabilities,
    "image_path": file_path
}

    return {
    "stage": stage,
    "confidence": confidence,
    "probabilities": probabilities,
    "info": assistant_data["info"],
    "precautions": assistant_data["precautions"],
    "diet": assistant_data["diet"],
    "advice": assistant_data["advice"]
}


@app.route("/download_report")
def download_report():

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    pdf = SimpleDocTemplate("Medical_Report.pdf")

    story = []

    # -----------------------------------
    # Title
    # -----------------------------------

    story.append(Paragraph(
        "AI-Based Diabetic Retinopathy Detection System",
        title_style
    ))

    story.append(Paragraph(
        "Medical Report",
        styles["Heading2"]
    ))

    story.append(Spacer(1,0.2*inch))

        # -----------------------------------
    # Uploaded Retinal Image
    # -----------------------------------

    story.append(
        Paragraph(
            "<b>Uploaded Retinal Image</b>",
            styles["Heading2"]
        )
    )

    try:

        img = PILImage.open(last_prediction["image_path"])

        img.thumbnail((250,250))

        temp_path = "temp_retina.jpg"

        img.save(temp_path)

        report_img = Image(temp_path)

        report_img.drawHeight = 2.8 * inch

        report_img.drawWidth = 2.8 * inch

        story.append(report_img)

    except Exception as e:

        story.append(
            Paragraph(
                "Image could not be loaded.",
                styles["Normal"]
            )
        )

    story.append(Spacer(1,0.3*inch))

    current_time = datetime.now().strftime("%d-%m-%Y %I:%M %p")

    story.append(
        Paragraph(
            f"<b>Generated On:</b> {current_time}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1,0.2*inch))

    # -----------------------------------
    # Summary Table
    # -----------------------------------

    risk = "Low"

    if "Mild" in last_prediction["stage"]:
        risk = "Mild"

    elif "Moderate" in last_prediction["stage"]:
        risk = "Moderate"

    elif "Severe" in last_prediction["stage"]:
        risk = "High"

    elif "Proliferative" in last_prediction["stage"]:
        risk = "Critical"

    data = [

        ["Diagnosis", last_prediction["stage"]],

        ["Confidence", f"{last_prediction['confidence']}%"],

        ["Risk Level", risk]

    ]

    table = Table(data, colWidths=[150,300])

    table.setStyle(TableStyle([

        ("GRID",(0,0),(-1,-1),1,colors.grey),

        ("BACKGROUND",(0,0),(0,-1),colors.lightblue),

        ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),

        ("BOTTOMPADDING",(0,0),(-1,-1),8)

    ]))

    story.append(table)

    story.append(Spacer(1,0.3*inch))

    # -----------------------------------
    # Probabilities
    # -----------------------------------

    story.append(
        Paragraph(
            "<b>Prediction Probabilities</b>",
            styles["Heading2"]
        )
    )

    probs = sorted(
        last_prediction["probabilities"].items(),
        key=lambda x:x[1],
        reverse=True
    )

    for label,value in probs:

        story.append(
            Paragraph(
                f"{label} : {value}%",
                styles["Normal"]
            )
        )

    story.append(Spacer(1,0.3*inch))

    # -----------------------------------
    # Disease Info
    # -----------------------------------

    story.append(
        Paragraph(
            "<b>Disease Information</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            last_prediction["info"],
            styles["Normal"]
        )
    )

    story.append(Spacer(1,0.2*inch))

    # -----------------------------------
    # Precautions
    # -----------------------------------

    story.append(
        Paragraph(
            "<b>Precautions</b>",
            styles["Heading2"]
        )
    )

    for item in last_prediction["precautions"]:

        story.append(
            Paragraph(
                "• "+item,
                styles["Normal"]
            )
        )

    story.append(Spacer(1,0.2*inch))

    # -----------------------------------
    # Diet
    # -----------------------------------

    story.append(
        Paragraph(
            "<b>Recommended Diet</b>",
            styles["Heading2"]
        )
    )

    for item in last_prediction["diet"]:

        story.append(
            Paragraph(
                "• "+item,
                styles["Normal"]
            )
        )

    story.append(Spacer(1,0.2*inch))

    # -----------------------------------
    # Advice
    # -----------------------------------

    story.append(
        Paragraph(
            "<b>Advice</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            last_prediction["advice"],
            styles["Normal"]
        )
    )

    story.append(Spacer(1,0.3*inch))

    # -----------------------------------
    # Disclaimer
    # -----------------------------------

    story.append(
        Paragraph(
            "<b>Medical Disclaimer</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            "This AI prediction is intended only for screening purposes and should not be considered a final medical diagnosis. Please consult a qualified ophthalmologist for confirmation.",
            styles["Italic"]
        )
    )

    story.append(Spacer(1,0.2*inch))

    story.append(
        Paragraph(
            "<b>Generated by AI-Based Diabetic Retinopathy Detection System</b>",
            styles["Normal"]
        )
    )
    story.append(Spacer(1,0.5*inch))

    story.append(
        Paragraph(
            "______________________________",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            "Authorized Medical Reviewer",
            styles["Normal"]
        )
    )

    pdf.build(story)

    return send_file(
        "Medical_Report.pdf",
        as_attachment=True
    )
# -----------------------------
# Main
# -----------------------------


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )