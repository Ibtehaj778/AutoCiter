from flask import Flask, request, jsonify, render_template
import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import PyPDF2
from tqdm import tqdm

app = Flask(__name__)
UPLOAD_FOLDER = '../papers/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model (ensure this is appropriate for your use case)
model = SentenceTransformer('all-MiniLM-L6-v2')

def read_pdf_files(pdf_paths):
    """Reads PDF files and extracts cleaned text line by line."""
    pdf_text_data = {}
    for pdf_path in pdf_paths:
        lines = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:  # Check if extract_text() returns a valid string
                        cleaned_text = re.sub(r'[^a-zA-Z0-9.,\s]', '', page_text)
                        lines += [line.strip() for line in re.split(r'[.,]', cleaned_text) if line.strip()]
                    else:
                        print(f"Warning: No readable text on a page in {pdf_path}")
            if lines:
                pdf_text_data[pdf_path] = lines
            else:
                print(f"Warning: No readable text found in PDF {pdf_path}")
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
    return pdf_text_data


def find_citation(query_line, pdf_text_data, top_k=2):
    """Finds the most relevant matches for a given query line across all PDFs."""
    sub_queries = [sub.strip() for sub in query_line.split('. ') if sub.strip()]
    all_top_matches = []

    for sub_query in sub_queries:
        query_embedding = model.encode(sub_query, convert_to_tensor=True)
        sub_query_matches = []

        for pdf_path, lines in pdf_text_data.items():
            try:
                line_embeddings = model.encode(lines, convert_to_tensor=True)
                similarity_scores = util.cos_sim(query_embedding, line_embeddings)[0]

                # Get top-k matches for the current sub-query
                top_indices = similarity_scores.topk(k=min(top_k, len(lines))).indices.tolist()
                for idx in top_indices:
                    score = similarity_scores[idx].item()
                    matched_line = lines[idx]
                    sub_query_matches.append((score, pdf_path, matched_line, sub_query))
            except Exception as e:
                print(f"Error processing PDF {pdf_path}: {e}")

        sub_query_matches = sorted(sub_query_matches, key=lambda x: x[0], reverse=True)[:top_k]
        all_top_matches.extend(sub_query_matches)

    # Convert to DataFrame for better structure
    all_top_matches_df = pd.DataFrame(all_top_matches, columns=['Similarity', 'Path', 'Line', 'Sub_query'])
    return all_top_matches_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    text = request.form['text']
    uploaded_files = request.files.getlist("pdf_files")
    pdf_paths = []

    # Save uploaded PDFs
    for file in uploaded_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        pdf_paths.append(file_path)
    
    # Read the PDF content
    pdf_text_data = read_pdf_files(pdf_paths)
    
    # Process each line in the input text
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    all_citations = []
    references = []
    ref_index = 1

    for line in lines:
        citations_df = find_citation(line, pdf_text_data, top_k=1)  # Top 1 match
        if not citations_df.empty:
            best_match = citations_df.iloc[0]
            ref_key = f"[{ref_index}]"
            all_citations.append(f"{line} {ref_key}")
            references.append(f"{ref_key} {best_match['Path']}: {best_match['Line']}")
            ref_index += 1
        else:
            all_citations.append(line)  # No citation found, keep original

    # Combine formatted lines and references
    formatted_text = ' '.join(all_citations)
    formatted_references = '\n'.join(references)

    return jsonify({
        'formatted_text': formatted_text,
        'references': formatted_references
    })

if __name__ == '__main__':
    app.run(debug=True)
