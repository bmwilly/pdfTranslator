import click

import PyPDF2
from fpdf import FPDF
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm


def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF file.
    """
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text


def translate_text(text, source_language, target_language, max_length=512):
    """
    Translates text from source language to target language.
    """
    # change lng
    model_name = f"Helsinki-NLP/opus-mt-{source_language}-{target_language}"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    text_segments = [text[i : i + max_length] for i in range(0, len(text), max_length)]

    translated_segments = []

    # Use tqdm for loading bar
    for segment in tqdm(text_segments, desc="Translating", unit="segment"):
        inputs = tokenizer(
            segment, return_tensors="pt", padding=True, max_length=max_length, truncation=True
        )
        outputs = model.generate(**inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated_segments.append(translated_text)

    translated_text = " ".join(translated_segments)
    return translated_text


def create_pdf_from_text(text, output_path):
    """
    Creates a PDF file from text.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 5, txt=text, align="L")
    pdf.output(output_path)


@click.command()
@click.option("--input_pdf_path", default="input.pdf", help="Input PDF file path.")
@click.option("--output_pdf_path", default="output.pdf", help="Output PDF file path.")
@click.option("--source_language", default="de", help="Source language.")
@click.option("--target_language", default="en", help="Target language.")
def process_pdf(input_pdf_path, output_pdf_path, source_language, target_language):
    text = extract_text_from_pdf(input_pdf_path)
    translated_text = translate_text(text, source_language, target_language)
    create_pdf_from_text(translated_text, output_pdf_path)


if __name__ == "__main__":
    process_pdf()
