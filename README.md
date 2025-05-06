# LLM-Embedding-and-LSTM

**Author:** Zhenlong Zhang

This app is used to link think-aloud embeddings with behavioral data using LSTM models. It supports both raw think-aloud text input (via OpenAI API) and precomputed embeddings to analyze alignment between language and behavior.

## APP Description
This app links think-aloud protocol embeddings with behavioral decision-making data using LSTM models.  
It enables users to:
- Upload behavior data, upload text embedding data, or upload think-aloud text
- Generate embeddings from text via OpenAI API or upload precomputed ones
- The model **text-embedding-ada-002** is used to generate text embedding
- Train both a baseline LSTM model and an LSTM model that incorporates text embeddings as its initial hidden state（H0)
- Perform Representational Similarity Analysis (RSA) between the text embeddings and the final hidden states of the base LSTM (The hidden state at the last trial should be sufficent as it integrates full trail history)
- Compare prediction accuracy in LSTM with and without embeddings

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Zhenlong-Zhang/LLM-Embedding-and-LSTM.git
cd LLM-Embedding-and-LSTM
```

### 2. (Optional) Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```
