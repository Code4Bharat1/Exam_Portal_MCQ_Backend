import os
import re
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pdf2image import convert_from_path
from PIL import Image
import requests
import json
from dotenv import load_dotenv

# === LOAD TOPIC MAP ===

with open("neet_topics.json", "r", encoding="utf-8") as f:
    TOPICS_MAP = json.load(f)

def get_topic_id(subject, chapter, topic_name):
    """Return topic_id for given subject, chapter, and topic_name, else None."""
    try:
        topics = TOPICS_MAP[subject][chapter]
        for topic in topics:
            if topic['name'].strip().lower() == topic_name.strip().lower():
                return topic['topic_id']
        # Optional: fuzzy fallback
        for topic in topics:
            if topic_name.strip().lower() in topic['name'].strip().lower():
                return topic['topic_id']
    except Exception:
        pass
    return None

# === GEMINI SETUP ===

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCA45LFI3V2w78b3FOeEvq0yrJe-VVm9VY")

GEMINI_EXTRACT_URL = (
    "https://generativelanguage.googleapis.com/v1/models/"
    f"gemini-1.5-pro:generateContent?key={API_KEY}" 
)

GEMINI_ASSESS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-1.5-pro-latest:generateContent?key={API_KEY}"
)

app = Flask(__name__)
CORS
# (app,origins=["*"]
#     #  ["https://superadmin-examportal.code4bharat.com/chapterwisequestion","http://localhost:3000"]
#     )

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# === HELPERS ===

def clean_html_tags(text):
    text = re.sub(r'<sub>|</sub>', '', text)
    text = re.sub(r'<sup>|</sup>', '', text)
    return text

def preprocess_image_for_mcq(image_path, output_path=None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception("Failed to load image for preprocessing")
    upscaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(upscaled, -1, kernel)
    contrasted = cv2.equalizeHist(sharpened)
    bw = cv2.adaptiveThreshold(contrasted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 25, 12)
    processed_path = output_path or (os.path.splitext(image_path)[0] + "_preprocessed.png")
    cv2.imwrite(processed_path, bw)
    return processed_path

def deskew_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception("Failed to load image for deskewing")
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    base, ext = os.path.splitext(image_path)
    corrected_path = f"{base}_deskewed{ext}"
    cv2.imwrite(corrected_path, rotated)
    return corrected_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "tiff"}

def allowed_image(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def extract_mcqs_from_image(image_path):
    base64_img = encode_image(image_path)
    prompt = """
You are an expert at extracting MCQs from academic exam pages.

From this image, extract ALL MCQs and return them as a JSON array.
For each MCQ, use the following keys:
- "question_number": the question number as an integer
- "question": the full question text
- "options": an array of 4 option texts, in order [(a), (b), (c), (d)]
- "answer": the correct option letter ("a", "b", "c", "d") if present, else ""

Example output:
[
  {
    "question_number": 192,
    "question": "Which one of the following is common to multicellular fungi, filamentous algae and protonema of mosses?",
    "options": [
      "Diplontic life cycle",
      "Members of kingdom plantae",
      "Mode of Nutrition",
      "Multiplication by fragmentation"
    ],
    "answer": "d"
  }
]
Do NOT return explanations. Do not include any markdown or commentary. Do not use code blocks.
If the correct answer is missing, set "answer" to "".
"""
    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": base64_img}},
                {"text": prompt}
            ]
        }]
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_EXTRACT_URL, headers=headers, json=payload)

    if response.ok:
        try:
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            # Remove code block wrappers if present
            if text.strip().startswith("```json"):
                text = text.strip().removeprefix("```json").removesuffix("```").strip()
            elif text.strip().startswith("```"):
                text = text.strip().removeprefix("```").removesuffix("```").strip()
            mcqs = json.loads(text)
            return mcqs
        except Exception as e:
            return [{"error": f"Parsing error: {str(e)}", "raw_text": text}]
    else:
        return [{"error": f"❌ Error {response.status_code}: {response.text}"}]

def assess_mcq_difficulty(mcq_full_text, chapter=None, topics=None):
    topics_str = ""
    if topics:
        topics_str = "\n".join([f"- {t}" for t in topics])

    prompt = f"""
You are an expert NEET MCQ evaluator and teacher.

Your tasks:
1. Assign a difficulty level to the MCQ using the NEET syllabus and pattern.
2. Identify the **correct answer**.
3. Provide a **clear explanation** that a NEET aspirant from standard 11 can easily understand.
"""

    if chapter and topics:
        prompt += f"""4. From the following list of topics for the chapter '{chapter}', pick the **single best-matching topic** for this MCQ. If none match, reply with 'Other'.

Topics:
{topics_str}
"""

    prompt += """
Return your response strictly in the following JSON format:

{
  "difficulty": "<easy | Medium | Hard>",
  "answer": "<Correct option letter: A/B/C/D>",
  "explanation": "<Clear and simple explanation>""" + (
        ',\n  "topic": "<Best matching topic from the list above or \'Other\'>"\n' if chapter and topics else '\n'
    ) + "}"

    prompt += """

Important notes:
- Use the difficulty rules as follows:
  - If the question is inherently **Easy** → output: easy
  - If the question is inherently **Medium** → output: Medium
  - If the question is inherently **Hard** → output: Hard
- Do not add extra commentary or formatting.
- Do not include any markdown or code blocks.
- Keep the explanation simple and NEET-appropriate.

MCQ:
""" + mcq_full_text

    headers = { "Content-Type": "application/json" }
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    response = requests.post(GEMINI_ASSESS_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = parts[0].get("text", "").strip() if parts else ""
    if text.startswith("```json"):
        text = text.replace("```json", "").strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        parsed = json.loads(text)
        difficulty = parsed.get("difficulty", "").lower()
        if difficulty not in ["easy", "medium", "hard"]:
            raise ValueError("Invalid difficulty")
        result = {
            "difficulty": difficulty,
            "answer": parsed.get("answer"),
            "explanation": parsed.get("explanation"),
        }
        if chapter and topics and "topic" in parsed:
            result["topic"] = parsed.get("topic")
        return result
    except Exception as e:
        raise ValueError(f"Invalid JSON response: {text}. Error: {e}")

# === ROUTES ===

@app.route("/api/extract-mcqs", methods=["POST"])
def extract_mcqs_api():
    # First check for PDF
    if 'pdf' in request.files and request.files['pdf'].filename != '':
        file = request.files['pdf']
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported for the 'pdf' field"}), 400
        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(pdf_path)
        try:
            pages = convert_from_path(pdf_path, dpi=300)
        except Exception as e:
            return jsonify({"error": f"PDF processing failed: {str(e)}"}), 500

        results = []
        for i, page in enumerate(pages):
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{os.path.splitext(file.filename)[0]}_page_{i+1}.png")
            page.save(image_path, "PNG")
            mcqs = extract_mcqs_from_image(image_path)
            results.extend(mcqs)  # flatten all MCQs

        return jsonify({"success": True, "type": "pdf", "mcqs": results})

    # Now check for image
    elif 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        if not allowed_image(file.filename):
            return jsonify({"error": f"Only image files are supported: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"}), 400
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)

        mcqs = extract_mcqs_from_image(image_path)
        return jsonify({"success": True, "type": "image", "mcqs": mcqs})

    else:
        return jsonify({"error": "No PDF or image file provided. Upload with key 'pdf' or 'image'."}), 400


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "MCQ API is running"}), 200

@app.route('/api/assess-difficulty', methods=['POST'])
def api_assess_difficulty():
    try:
        data = request.get_json()
        mcq = data.get("mcq")
        chapter = data.get("chapter")
        topics = data.get("topics")
        subject = data.get("subject")  # <-- ADDED

        if not mcq:
            return jsonify({"error": "MCQ question text (including options) is required"}), 400

        option_patterns = [
            r'(\(a\)|a\.)',
            r'(\(b\)|b\.)',
            r'(\(c\)|c\.)',
            r'(\(d\)|d\.)',
        ]
        if not all(re.search(pat, mcq, re.IGNORECASE) for pat in option_patterns):
            return jsonify({
                "error": "MCQ must contain all 4 options: (a), (b), (c), (d) or A., B., etc."
            }), 400

        result = assess_mcq_difficulty(mcq, chapter, topics)

        # --- MAP TOPIC TO ID IF POSSIBLE ---
        if subject and chapter and "topic" in result:
            topic_id = get_topic_id(subject, chapter, result["topic"])
            result["topic_id"] = topic_id  # could be None if not found

        return jsonify(result)
    except requests.exceptions.RequestException as e:   
        return jsonify({"error": "API request error", "details": str(e)}), 500
    except ValueError as e:
        return jsonify({"error": "Invalid response", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Unexpected error", "details": str(e)}), 500

if __name__ == "__main__":
    from waitress import serve
    print("the server is running on port 603")
    serve(app, host="0.0.0.0", port=6003)
