import streamlit as st
import pandas as pd
import os
import numpy as np

from generate_embedding import generate_all_embeddings_from_df
from analysis_tools import train_base_and_extract_hidden, train_enhanced_and_compare, run_rsa, plot_accuracy_comparison
from model_definitions import BaseLSTM, LSTMwithEmbedding
from dataset_utils import BehaviorDataset

st.set_page_config(page_title="Think-Aloud + Behavior App", layout="wide")
st.title("Linking Think-Aloud Embeddings with Behavior")


with st.expander("Click to View Example File Formats"):
    try:
        behavior_example = pd.read_csv("example-format/example_behavior.csv")
        st.markdown("#### Example Behavior CSV Format")
        st.dataframe(behavior_example)

        embedding_example = pd.read_csv("example-format/example_embedding.csv")
        st.markdown("#### Example Embedding CSV Format")
        st.dataframe(embedding_example)

        text_example = pd.read_csv("example-format/example_text.csv")
        st.markdown("#### Example Text CSV Format (for OpenAI API)")
        st.dataframe(text_example)
    except Exception as e:
        st.error(f" Error loading example files: {e}")
# -------------------------
# two options to choose
# -------------------------
upload_mode = st.radio(
    "Select your embedding input type:",
    [" I have embedding CSV", " I have raw think-aloud text CSV"]
)

# -------------------------
# behavior data
# -------------------------
behavior_file = st.file_uploader(" Upload Behavior Data (.csv)", type=["csv"])
embedding_df = None

# -------------------------
# optional 1: with embedding CSV
# -------------------------
if upload_mode == " I have embedding CSV":
    embedding_file = st.file_uploader(" Upload Embedding CSV", type=["csv"])
    if behavior_file and embedding_file:
        behavior_df = pd.read_csv(behavior_file)
        embedding_df = pd.read_csv(embedding_file)
        st.success(" Behavior and embedding CSV files uploaded.")

# -------------------------
# option 2: text  with API key
# -------------------------
elif upload_mode == " I have raw think-aloud text CSV":
    text_file = st.file_uploader(" Upload Think-Aloud Text CSV ", type=["csv"])
    api_key = st.text_input(" Enter your OpenAI API Key", type="password")

    if behavior_file and text_file and api_key:
        behavior_df = pd.read_csv(behavior_file)
        text_df = pd.read_csv(text_file)
        st.write("Text CSV Preview:")
        st.dataframe(text_df.head())

        if st.button(" Generate Embedding from Text"):
            with st.spinner("Generating embeddings using OpenAI..."):
                output_path = "data/generated_embeddings.csv"
                os.makedirs("data", exist_ok=True)
                embedding_df = generate_all_embeddings_from_df(text_df, api_key)
                embedding_df.to_csv(output_path, index=False)

                st.success("✅ Embedding generated and loaded.")

# -------------------------
# behavioral LSTM, analyse hidden states with text embedding doing RSA. and if sign, use embedding as h0
# -------------------------
if behavior_file is not None and embedding_df is not None:

    
    st.session_state.model_class_base = BaseLSTM
    st.session_state.model_class_h0   = LSTMwithEmbedding
    st.session_state.dataset_base     = BehaviorDataset(behavior_df)
    st.session_state.dataset_h0       = BehaviorDataset(behavior_df, embedding_df, use_h0=True)



    # Preview
    st.write("### Behavior Data Preview")
    st.dataframe(behavior_df.head())
    st.write("### Embedding Data Preview")
    st.dataframe(embedding_df.head())

    #  Fit base LSTM & extract hidden
    if st.button(" Fit Base LSTM & Extract Hidden"):
        with st.spinner("Training base LSTM and extracting hidden states…"):
            model_base, hidden_states, block_ids = train_base_and_extract_hidden(
                model_class=st.session_state.model_class_base,
                dataset=st.session_state.dataset_base,
                hidden_dim=32,
                epochs=10
            )
            
            st.session_state.model_base = model_base
            st.session_state.hidden_states = hidden_states

            reduced_embeds = np.stack([v.numpy() if hasattr(v, "numpy") else v for v in st.session_state.dataset_h0.h0_vecs])
            st.session_state.embedding_matrix = reduced_embeds

            st.session_state.hidden_dict = {bid: vec for bid, vec in zip(block_ids, hidden_states)}
            st.session_state.embedding_dict = {bid: vec for bid, vec in zip(block_ids, reduced_embeds)}

        st.success("✅ Base LSTM trained and hidden states extracted.")

    # RSA analysis
    if "hidden_dict" in st.session_state and "embedding_dict" in st.session_state:
        if st.button(" Run RSA Analysis"):
            with st.spinner("Running RSA analysis…"):
                common_ids = sorted(set(st.session_state.hidden_dict) & set(st.session_state.embedding_dict))
                hidden_mat = np.stack([st.session_state.hidden_dict[i] for i in common_ids])
                embed_mat  = np.stack([st.session_state.embedding_dict[i] for i in common_ids])
                r_val, p_val = run_rsa(hidden_mat, embed_mat)  
            st.success(f"✅ RSA completed: Spearman r = {r_val:.3f}, p = {p_val:.4f}")

    #  Accuracy comparison (cross-validation)
    if "hidden_states" in st.session_state:
        if st.button(" Show Accuracy Comparison"):
            with st.spinner("Comparing base vs enhanced model…"):
                acc_base, acc_h0 = train_enhanced_and_compare(
                    model_class_base=st.session_state.model_class_base,
                    model_class_h0=st.session_state.model_class_h0,
                    dataset_base=st.session_state.dataset_base,
                    dataset_h0=st.session_state.dataset_h0,
                    emb_dim=st.session_state.embedding_matrix.shape[1],
                    epochs=10,
                    k=5
                )

                plot_accuracy_comparison(acc_base, acc_h0)
            st.success("✅ Accuracy comparison done.")

else:
    st.info("Please upload all required files to proceed.")
