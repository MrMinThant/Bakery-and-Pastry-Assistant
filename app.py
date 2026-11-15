
import os
from dotenv import load_dotenv

load_dotenv()
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# Connect to Gemini using your API key
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Set your API key in the terminal."
    )

client = genai.Client(api_key=api_key)

# KNOWLEDGE INJECTION PROMPT
KNOWLEDGE_INJECTION_PROMPT = """
You are an expert Bakery and Pastry Identification AI Assistant.

Your job is to identify bakery products from images
and answer questions spoken by the user.

CLASSIFICATION KNOWLEDGE:

1. WHITE BREAD:
- Pale ivory or cream-colored inside.
- Fine, soft and relatively uniform air pockets.
- Usually has a smooth golden-brown crust.
- No obvious whole-grain flecks.

2. WHOLE WHEAT BREAD:
- Tan to medium-brown inside.
- Often has a denser texture.
- Visible brown bran flecks or grain particles.
- May contain visible oats or seeds.

3. FLAKY PASTRY / CROISSANT:
- Golden-brown exterior.
- Visible thin, crispy pastry layers.
- Honeycomb-like air pockets inside.
- Often crescent-shaped or folded.

INSTRUCTIONS:

1. Examine the uploaded bakery image.
2. Listen to and understand the user's audio question.
3. Classify the item into ONE category:
   White Bread, Whole Wheat Bread,
   Flaky Pastry / Croissant, or Unknown.
4. Explain the visual evidence for your classification.
5. Answer the user's spoken question clearly.
6. Do not guess if the image is unclear.
7. Do not claim nutritional values from appearance alone.

RESPONSE FORMAT:

Identified Product:
Category:
Visual Evidence:
Answer to User:
Confidence: High / Medium / Low
"""

# HOME PAGE
@app.route("/")
def index():
    return render_template("index.html")


# IMAGE + AUDIO ANALYSIS
@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files or "audio" not in request.files:
        return jsonify({
            "error": "Please upload an image and record a voice question."
        }), 400

    image_file = request.files["image"]
    audio_file = request.files["audio"]

    if not image_file.filename or not audio_file.filename:
        return jsonify({
            "error": "One of the uploaded files is empty."
        }), 400

    try:
        image_bytes = image_file.read()
        audio_bytes = audio_file.read()

        if not image_bytes or not audio_bytes:
            return jsonify({
                "error": "The image or audio recording is empty."
            }), 400

        image_mime = image_file.mimetype or "image/jpeg"
        audio_mime = audio_file.mimetype or "audio/webm"

        # Send BOTH inputs and the knowledge prompt to Gemini
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=image_mime
                ),
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=audio_mime
                ),
                "Identify the bakery product in the image "
                "and answer the spoken question."
            ],
            config=types.GenerateContentConfig(
                system_instruction=KNOWLEDGE_INJECTION_PROMPT,
                temperature=0.2
            )
        )

        return jsonify({
            "answer": response.text
        })

    except Exception as error:
        app.logger.exception("Gemini analysis failed")
        return jsonify({
            "error": "AI analysis failed. Please check the server terminal."
        }), 500


# START THE SERVER
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
