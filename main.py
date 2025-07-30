# main.py
import os
import argparse
import pymupdf
from dotenv import load_dotenv
from PodcastGenerator import PodcastGenerator


def read_pdf(file_path):
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

def read_pdf_with_structure(file_path):
    """Extract PDF with font size information to identify headers."""
    doc = pymupdf.open(file_path)
    text_with_structure = ""
    
    for page in doc:
        blocks = page.get_text("dict")
        
        for block in blocks["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    max_font_size = 0
                    
                    for span in line["spans"]:
                        line_text += span["text"]
                        max_font_size = max(max_font_size, span["size"])
                    
                    line_text = line_text.strip()
                    if line_text:
                        # Convert larger fonts to markdown headers
                        if max_font_size > 14:  # Adjust threshold as needed
                            text_with_structure += f"# {line_text}\n"
                        elif max_font_size > 12:
                            text_with_structure += f"## {line_text}\n"
                        else:
                            text_with_structure += f"{line_text}\n"
                        
                text_with_structure += "\n"
    
    return text_with_structure


def main():
    parser = argparse.ArgumentParser(description='Generate a podcast from a document')
    parser.add_argument('--input', type=str, required=True, help='Path to input document')
    parser.add_argument('--output', type=str, default='output_podcast.wav', help='Path to output audio file')
    parser.add_argument('--model', type=int, choices=[1, 2], default=1, help='Select TTS model: 1 for Kokoro, 2 for SageMaker')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for more verbose output')
    args = parser.parse_args()
    
    load_dotenv()

    try:
        file_extension = os.path.splitext(args.input)[1].lower()
        if file_extension == '.pdf':
            document = read_pdf_with_structure(args.input)
            if args.debug:
                print(f"[DEBUG] Extracted text preview:\n{document[:500]}...")
        else:
            with open(args.input, 'r', encoding='utf-8') as f:
                document = f.read()
        
        podcast_gen = PodcastGenerator()
        model_name = 'kokoro' if args.model == 1 else 'sagemaker'
        print(f"Using model: {model_name}")

        output_file = podcast_gen.generate_podcast(document, args.output, model_name=model_name)
        print(f"Podcast generated successfully: {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
