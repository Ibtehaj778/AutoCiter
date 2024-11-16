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
    pdf_text_data = {}
    for pdf_path in pdf_paths:
        lines = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    cleaned_text = re.sub(r'[^a-zA-Z0-9.,\s]', '', page_text)
                    lines += [line.strip() for line in re.split(r'[.,]', cleaned_text) if line.strip()]
        pdf_text_data[pdf_path] = lines
    return pdf_text_data

def find_citation(query_line, pdf_text_data, top_k=2):
    sub_queries = [sub.strip() for sub in query_line.split('. ') if sub.strip()]
    all_top_matches = []

    for sub_query in sub_queries:
        query_embedding = model.encode(sub_query, convert_to_tensor=True)
        sub_query_matches = []

        for pdf_path, lines in tqdm(pdf_text_data.items()):
            line_embeddings = model.encode(lines, convert_to_tensor=True)
            similarity_scores = util.cos_sim(query_embedding, line_embeddings)[0]
            
            top_indices = similarity_scores.topk(k=top_k).indices.tolist()
            
            for idx in top_indices:
                score = similarity_scores[idx].item()
                matched_line = lines[idx]
                sub_query_matches.append((score, pdf_path, matched_line, sub_query))
        
        sub_query_matches = sorted(sub_query_matches, key=lambda x: x[0], reverse=True)[:top_k]
        all_top_matches.extend(sub_query_matches)
    
    all_top_matches_df = pd.DataFrame(all_top_matches, columns=['Similarity', 'Path', 'Line', 'Sub_query'])
    return all_top_matches_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Handle file upload
    uploaded_files = request.files.getlist('pdfFiles')
    query_text = request.form.get('query')
    
    if not uploaded_files or not query_text:
        return jsonify({"error": "Missing files or query"}), 400

    pdf_paths = []
    for file in uploaded_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        pdf_paths.append(file_path)
    
    # Process PDFs and query
    try:
        pdf_text_data = read_pdf_files(pdf_paths)
        matches_df = find_citation(query_text, pdf_text_data)
        return jsonify(matches_df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

