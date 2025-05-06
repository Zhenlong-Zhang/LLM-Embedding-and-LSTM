# LLM-Embedding-and-LSTM

**Author:** Zhenlong Zhang

This app is used to link think-aloud embeddings with behavioral data using LSTM models. It supports both raw think-aloud text input (via OpenAI API) and precomputed embeddings to analyze alignment between language and behavior.

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Zhenlong-Zhang/LLM-Embedding-and-LSTM.git
cd LLM-Embedding-and-LSTM

### 2. (Optional) Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

### 3. Install required packages
pip install -r requirements.txt

### 4. Launch the Streamlit app
streamlit run app.py

