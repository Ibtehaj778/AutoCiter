# AutoCiter

AutoCiter is a Flask-based tool that uses sentence transformers and cosine similarity to automatically cite text from uploaded PDF papers.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Ibtehaj778/AutoCiter.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the backend:

   ```bash
   python backend.py
   ```

   This tool works on MPS, CUDA, and CPU.

## Usage

1. Open your browser and go to `http://127.0.0.1:5000/`.
2. Upload your PDF papers.
3. Once the PDFs are uploaded, add the text you want to analyze.
4. Click on "Analyze". The tool will process the text and provide all possible references for each line.

## YouTube Tutorial

For a step-by-step tutorial, check out the YouTube video by clicking the image below:

[![AutoCiter YouTube Tutorial](https://img.youtube.com/vi/XhESgjRFor0/0.jpg)](https://youtu.be/XhESgjRFor0)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
