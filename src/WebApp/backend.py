from flask import Flask, request, jsonify, render_template
import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import PyPDF2
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch
from flask import Response, stream_with_context
import time

device = None
if torch.cuda.is_available():
    device = 'cuda'
elif torch.has_mps:
    device = 'mps'
else:
    device='cpu'

app = Flask(__name__)
UPLOAD_FOLDER = '../papers/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model (ensure this is appropriate for your use case)
model = SentenceTransformer('all-MiniLM-L6-v2').to(device=device)

def preprocess_pdf_text(pdf_path):
    lines = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                cleaned_text = re.sub(r'[^a-zA-Z0-9.,\s]\n', '', page_text)
                lines += [line.strip() for line in re.split(r'[.,]', cleaned_text) if line.strip()]
    return lines

def precompute_pdf_embeddings(pdf_paths, save_path):
    embedding_data = []
    global progress_log

    for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
        progress_log.append(f"Processing {os.path.basename(pdf_path)}...")
        lines = preprocess_pdf_text(pdf_path)
        line_embeddings = model.encode(lines, convert_to_tensor=True).cpu()
        for idx, (line, embedding) in enumerate(zip(lines, line_embeddings)):
            embedding_data.append({
                'Path': pdf_path,
                'Line': line,
                'Embedding': embedding.numpy()
            })

    df_embeddings = pd.DataFrame(embedding_data)
    df_embeddings.to_pickle(save_path)
    progress_log.append(f"Embeddings saved to {save_path}")

@app.route('/precompute_pdf_embeddings', methods=['POST'])
def handle_pdf_embedding_request():
    try:
        # Get uploaded files
        pdf_files = request.files.getlist('pdf_files')
        if not pdf_files:
            return jsonify({"error": "No PDF files uploaded"}), 400

        # Save the PDFs temporarily for processing
        temp_dir = './uploads/'
        os.makedirs(temp_dir, exist_ok=True)

        pdf_paths = []
        for pdf in pdf_files:
            pdf_path = os.path.join(temp_dir, pdf.filename)
            pdf.save(pdf_path)
            pdf_paths.append(pdf_path)

        # Define the save path for the embeddings
        save_path = './embedding_data.pkl'

        # Precompute the embeddings
        precompute_pdf_embeddings(pdf_paths, save_path)

        # Send success response
        return jsonify({"message": "Embeddings precomputed successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def load_embeddings(embedding_path):
    df_embeddings = pd.read_pickle(embedding_path)
    df_embeddings['Embedding'] = df_embeddings['Embedding'].apply(torch.tensor)
    return df_embeddings

def find_citation_with_embeddings(query, df_embeddings):
    df_embeddings['Embedding'] = df_embeddings['Embedding'].apply(
        lambda x: x.clone().detach().to(device) if isinstance(x, torch.Tensor) else torch.tensor(x).to(device)
    )
    query_lines = sent_tokenize(query)
    results = []
    for query_line in tqdm(query_lines, desc="Processing query lines"):
        query_embedding = model.encode(query_line, convert_to_tensor=True).to(device)
        scores = [
            (util.cos_sim(query_embedding, row.Embedding).item(), row.Path, row.Line)
            for row in df_embeddings.itertuples(index=False)
        ]
        best_match = max(scores, key=lambda x: x[0])
        results.append((query_line, *best_match))
    
    return pd.DataFrame(results, columns=["Sub Query", "Similarity", "Path", "Line"])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    text = request.form['text']
    uploaded_files = request.files.getlist("pdf_files")
    pdf_paths = []

    text = text.replace('\n', ' ')
    text = text.strip()
    for file in uploaded_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        pdf_paths.append(file_path)
        
    # Load precomputed embeddings
    pdf_text_data = load_embeddings("embeddings.pkl")

    lines = [line.strip() for line in text.split('. ') if line.strip()]
    formatted_lines = []
    references = {}
    ref_index = 1

    i = 0
    for line in lines:
        citations_df = find_citation_with_embeddings(line, pdf_text_data)  # Retrieve all matches
        i += 1
        citations_df['Path'] = citations_df['Path'].apply(lambda x: os.path.basename(x))
        paths = citations_df['Path'].unique()
        for path in paths:
            references[path] = ref_index
            ref_index += 1

        progress_log.append(f"Processing Line #{i}")
        if not citations_df.empty:
            for _, match in citations_df.iterrows():
                ref_key = f"[{references[os.path.basename(match['Path'])]}]"
                if line[-1] == '.':
                    line = line[0:len(line)-1]
                elif line[-1:-2] == '. ':
                    line = line[0:len(line)-1]
                
                formatted_lines.append(f"{line} {ref_key}")
        else:
            formatted_lines.append(line)

    progress_log.append(f"Processing Complete!!!")
    formatted_text = '\n'.join(formatted_lines)
    formatted_references = '\n'.join([f"[{value}] {key}" for key, value in references.items()])

    return jsonify({
        'formatted_text': formatted_text,
        'references': formatted_references
    })

progress_log = []

@app.route('/progress')
def progress():
    def generate():
        while True:
            if progress_log:
                yield f"data: {progress_log.pop(0)}\n\n"
            time.sleep(0.1)  # Adjust this for real-time updates

    return Response(stream_with_context(generate()), content_type='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)
