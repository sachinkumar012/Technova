from flask import Flask, render_template, request
import PyPDF2
from transformers import pipeline
import os

app = Flask(__name__)

# Load the AI model for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    if request.method == "POST":
        uploaded_file = request.files["file"]
        if uploaded_file.filename.endswith(".pdf"):
            pdf_path = os.path.join("uploads", "uploaded.pdf")
            uploaded_file.save(pdf_path)
            text = extract_text_from_pdf(pdf_path)
            summary = summarizer(text[:1024])[0]['summary_text']  # Summarize only first 1024 characters
    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
