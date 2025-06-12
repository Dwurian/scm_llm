import PyPDF2
import os
import fitz
from collections import Counter
import tiktoken

#########################
### Create Text Files ###
#########################

def pdf_to_text(pdf_path):
    # Open the PDF file in binary mode.
    with open(pdf_path, 'rb') as file:
        # Create a PdfReader object.
        reader = PyPDF2.PdfReader(file)
        # Initialize an empty string for the text.
        text = ""
        # Iterate through all pages and extract text.
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

if __name__ == "__main__":
    pdf_dir = r"graph_db/demo3/pdfs"
    txt_dir = r"graph_db/demo3/txt"

    os.makedirs(txt_dir, exist_ok=True)

    # Loop over all files in the directory.
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            extracted_text = pdf_to_text(pdf_path)

            # Define the txt file name and path.
            txt_file_name = os.path.splitext(pdf_file)[0] + ".txt"
            txt_file_path = os.path.join(txt_dir, txt_file_name)

            # Save the extracted text into the txt file.
            with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(extracted_text)

            print(f"Saved text for {pdf_file} to {txt_file_name}")

#######################
### Extract Abstract###
#######################

def extract_abstract(text):
    # Convert the text to lowercase to ensure case-insensitive matching.
    text_lower = text.lower()

    # Find the start index for the abstract.
    abs_index = text_lower.find("abstract")
    if abs_index == -1:
        return None  # If "abstract" isn't found, return None.

    # Set the start index just after the word "abstract".
    start = abs_index + len("abstract")

    # Define possible markers that might indicate the end of the abstract.
    end_markers = ["introduction", "background", "overview", "intro", "opening", "prolog"]

    # Find the earliest occurrence of any of the end markers after the abstract.
    end_indices = [text_lower.find(marker, start) for marker in end_markers if text_lower.find(marker, start) != -1]

    if end_indices:
        end_index = min(end_indices)
        abstract = text[start:end_index]
    else:
        # If none of the markers are found, take everything from "abstract" to the end.
        abstract = text[start:]

    return abstract.strip()

if __name__ == "__main__":
    # Directory containing the text files of academic papers.
    txt_dir = r"graph_db/demo3/txt"
    # Directory where the extracted abstract files will be saved.
    chunk_dir = r"graph_db/demo3/chunk"

    # Create the output directory if it doesn't already exist.
    os.makedirs(chunk_dir, exist_ok=True)

    # Process every text file in the input directory.
    for txt_file in os.listdir(txt_dir):
        if txt_file.lower().endswith(".txt"):
            file_path = os.path.join(txt_dir, txt_file)

            base_name = os.path.splitext(txt_file)[0]
            paper_dir = os.path.join(chunk_dir, base_name)

            os.makedirs(paper_dir, exist_ok=True)

            # Read the full content of the text file.
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            # Extract the abstract chunk from the full text.
            abstract_text = extract_abstract(full_text)

            # Define the output text file name (e.g. paper_abstract.txt).
            output_file = os.path.splitext(txt_file)[0] + "_abstract.txt"
            output_path = os.path.join(paper_dir, output_file)

            if abstract_text:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(abstract_text)
                print(f"Abstract for '{txt_file}' saved to '{output_file}'.")
            else:
                print(f"No abstract found in '{txt_file}'.")

####################################
#### Chunk Sections of Interest ####
####################################

def extract_section_from_pdf(pdf_path, section_keyword):
    doc = fitz.open(pdf_path)
    found_section = False
    section_text = ""
    heading_size = None
    headings = []
    font_combos = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span["size"]  # rounded for grouping
                    font = span["font"]           # typeface (e.g., "TimesNewRomanPSMT")
                    flags = span["flags"]         # numeric font flags
                    combo = (size, font, flags)
                    font_combos.append(combo)

    font_counter = Counter(font_combos)
    body_font = font_counter.most_common(1)
    (body_size, body_font, body_flags), count = body_font[0]
    #print(f"Most common font combination: {body_size, body_font, body_flags} (occurs {count} times)")

    # Iterate through pages in order.
    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    span_text = span["text"].strip()
                    if not span_text:
                        continue

                    is_heading = False

                    text = span["text"].strip()
                    font_size = span["size"]
                    font = span["font"]
                    font_flags = span["flags"]
                    is_heading = (font_size >= body_size and
                                  (font != body_font or font_flags != body_flags) and
                                  (len(text) > 2 and
                                   not text.islower()))

                    if not found_section:
                        # Search for the section heading
                        if (section_keyword.lower() in span_text.lower() and is_heading):
                            found_section = True
                            heading_size = span["size"]
                            # Optionally, skip adding the heading text itself.
                            continue
                    else:
                        # Once the section heading is found, add subsequent text spans
                        # if they appear in a smaller font than the heading.
                        if span["size"] < heading_size:
                            section_text += span_text + " "
                        else:
                            # If we encounter a span with a font size that is equal
                            # or larger than the heading, assume the section ended.
                            return section_text.strip()
    return section_text.strip()

def chunk_text_by_tokens(text, max_tokens=5000, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end

    return chunks

if __name__ == "__main__":
    # Your provided paths:
    pdf_dir = r"graph_db/demo3/pdfs"
    chunk_dir = r"graph_db/demo3/chunk"

    # Define the sections we want to extract.
    # The keys are the keywords to search for (in headings) and the values
    # are the subfolder names and file suffixes for the output.
    sections = {
        "introduction": {"folder": "intro", "suffix": "intro"},
        "conclu": {"folder": "concl", "suffix": "concl"},
        "literature": {"folder": "liter", "suffix": "liter"},  # For literature review sections
        "reference": {"folder": "ref", "suffix": "ref"},
    }

    # Process every PDF in the pdf_dir.
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            base_name = os.path.splitext(pdf_file)[0]
            for keyword, info in sections.items():
                section_content = extract_section_from_pdf(pdf_path, keyword)
                if section_content:

                    # Chunk further if necessary
                    chunks = chunk_text_by_tokens(section_content, max_tokens=5000,
                                                  model = "text-embedding-ada-002")

                    # Create a subfolder for the section if it doesn't exist.
                    paper_folder = os.path.join(chunk_dir, base_name)
                    os.makedirs(paper_folder, exist_ok=True)

                    output_file = f"{base_name}_{info['suffix']}"

                    if len(chunks) == 1:
                        file_path = os.path.join(paper_folder, f"{output_file}.txt")
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(chunks[0])
                        print(f"{keyword.capitalize()} extracted from {pdf_file} saved to {output_file} in folder {base_name}")
                    else:
                        for i, chunk in enumerate(chunks):
                            output_file_new = f"{output_file}_{i + 1}"
                            file_path = os.path.join(paper_folder, f"{output_file_new}.txt")
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(chunk)
                            print(f"{keyword.capitalize()} extracted from {pdf_file} saved to {output_file} in folder {base_name}")
                else:
                    print(f"{keyword.capitalize()} not found in {pdf_file}")

