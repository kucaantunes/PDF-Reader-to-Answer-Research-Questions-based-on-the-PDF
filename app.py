import os
import re
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load models and tokenizers
MODEL_NAMES = {
    'bart': "facebook/bart-large-cnn",
    'gpt2': "gpt2",
    'gpt_neo': "EleutherAI/gpt-neo-2.7B",
}

tokenizers = {
    'bart': AutoTokenizer.from_pretrained(MODEL_NAMES['bart']),
    'gpt2': AutoTokenizer.from_pretrained(MODEL_NAMES['gpt2']),
    'gpt_neo': AutoTokenizer.from_pretrained(MODEL_NAMES['gpt_neo']),
}

models = {
    'bart': AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAMES['bart']),
    'gpt2': AutoModelForCausalLM.from_pretrained(MODEL_NAMES['gpt2']),
    'gpt_neo': AutoModelForCausalLM.from_pretrained(MODEL_NAMES['gpt_neo']),
}

def extract_references(pdf_text):
    """Extract references from the bottom of the text based on common citation patterns."""
    references_pattern = r"(\[\d+\]\s.+?)(?=\n\n|\Z)"
    matches = re.findall(references_pattern, pdf_text, re.MULTILINE | re.DOTALL)
    return matches if matches else ["No references found."]

def generate_answer(model_name, pdf_text, question, max_length=1500, min_length=700):
    """Generate an answer using the specified model."""
    try:
        model = models[model_name]
        tokenizer = tokenizers[model_name]

        input_text = f"Research Question: {question}\n\nContext: {pdf_text}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)

        outputs = model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=5,
            length_penalty=1.5,
            early_stopping=True
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        logging.error(f"Error generating answer with {model_name}: {e}")
        return f"Error generating answer: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    question = request.form.get('question')
    pdf_text = request.form.get('pdf_text')
    uploaded_file = request.files.get('file')

    if not question or not (pdf_text or uploaded_file):
        return jsonify({"error": "Both question and PDF text are required"}), 400

    if uploaded_file and allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(filepath)
        pdf_text = "Simulated text from PDF extraction."

    # Extract references
    references = extract_references(pdf_text)

    # Generate answers
    bart_answer = generate_answer('bart', pdf_text, question)
    gpt2_answer = generate_answer('gpt2', pdf_text, question)
    gpt_neo_answer = generate_answer('gpt_neo', pdf_text, question)

    return render_template('result.html',
                           question=question,
                           bart_answer=bart_answer,
                           gpt2_answer=gpt2_answer,
                           gpt_neo_answer=gpt_neo_answer,
                           references=references)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
