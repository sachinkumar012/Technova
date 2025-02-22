import PyPDF2
import textwrap
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_txt(txt_path):
    """Extracts text from a TXT file."""
    with open(txt_path, "r", encoding="utf-8") as file:
        return file.read()

def chunk_text(text, chunk_size=1024):
    """Splits text into smaller chunks for processing."""
    return textwrap.wrap(text, width=chunk_size)

def summarize_text(text):
    """Summarizes the given text using a pre-trained AI model."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    chunks = chunk_text(text)
    summary = ""
    for chunk in chunks:
        summary += summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]["summary_text"] + "\n"
    return summary

def main(file_path):
    """Main function to summarize a book."""
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".txt"):
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or TXT.")
    
    summary = summarize_text(text)
    print("\nSummary:\n", summary)

if __name__ == "__main__":
    file_path = input("Enter the path to the book file (PDF or TXT): ")
    main(file_path)
