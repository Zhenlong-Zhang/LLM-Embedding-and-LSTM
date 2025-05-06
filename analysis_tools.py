import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import streamlit as st

def train_base_and_extract_hidden(model_class, dataset, hidden_dim=32, epochs=10, batch_size=8, lr=0.01, device="cpu"):
    # initialize model
    model = model_class(input_size=2, hidden_size=hidden_dim, output_size=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # training loop
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    # extract hidden states
    model.eval()
    hidden_states = []
    block_ids = dataset.block_ids  
    infer_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in infer_loader:
            x, _ = batch
            x = x.to(device)
            out, _ = model.lstm(x)
            h = out[:, -1, :].squeeze(0).cpu().numpy()
            hidden_states.append(h)

    hidden_states = np.stack(hidden_states)
    return model, hidden_states, block_ids



def train_enhanced_and_compare(model_class_base, model_class_h0, dataset_base, dataset_h0,
                               emb_dim, hidden_dim=32, epochs=10, batch_size=8,
                               lr=0.01, device="cpu", k=5):

    acc_base = run_crossval_accuracy(
        model_class=model_class_base,
        dataset=dataset_base,
        use_h0=False,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        k=k
    )

    acc_h0 = run_crossval_accuracy(
        model_class=model_class_h0,
        dataset=dataset_h0,
        use_h0=True,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        k=k
    )

    return acc_base, acc_h0


def run_crossval_accuracy(model_class, dataset, use_h0=False, emb_dim=None,
                          hidden_dim=32, batch_size=8, lr=0.01, device="cpu",
                          k=5, epochs=10):
    accs = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(dataset):
        train_ds = Subset(dataset, train_idx)
        test_ds = Subset(dataset, test_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        if use_h0:
            model = model_class(input_size=2, hidden_size=hidden_dim,
                                output_size=2, emb_dim=emb_dim).to(device)
        else:
            model = model_class(input_size=2, hidden_size=hidden_dim,
                                output_size=2).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                if use_h0:
                    x, h0, y = batch
                    x, h0, y = x.to(device), h0.to(device), y.to(device)
                    pred = model(x, h0)
                else:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                if use_h0:
                    x, h0, y = batch
                    x, h0 = x.to(device), h0.to(device)
                    pred = model(x, h0)
                else:
                    x, _y = batch
                    x = x.to(device)
                    pred = model(x)
                preds.extend(torch.argmax(pred, dim=1).cpu().numpy())
                trues.extend(y.cpu().numpy())
        accs.append(accuracy_score(trues, preds))

    return accs



def plot_accuracy_comparison(acc_base, acc_h0, output_path=None):
    import matplotlib.pyplot as plt
    import streamlit as st

    plt.figure(figsize=(6, 4))
    plt.plot(acc_base, label='Base LSTM (no embedding)', marker='o')
    plt.plot(acc_h0, label='Enhanced LSTM (embedding as h0)', marker='s')
    plt.title("Cross-Validation Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)

    st.pyplot(plt.gcf())  
    plt.close()




def run_rsa(hidden_mat, embed_mat):
    if hidden_mat.shape[0] != embed_mat.shape[0]:
        raise ValueError(f"Input size mismatch: hidden_mat has {hidden_mat.shape[0]} rows, "
                         f"embed_mat has {embed_mat.shape[0]} rows.")

    # Compute cosine similarity matrices
    sim_hidden = cosine_similarity(hidden_mat)
    sim_embed = cosine_similarity(embed_mat)

    # Flatten upper triangles for correlation
    flat_h = sim_hidden[np.triu_indices_from(sim_hidden, k=1)]
    flat_e = sim_embed[np.triu_indices_from(sim_embed, k=1)]
    r, p = spearmanr(flat_h, flat_e)

    # --- Plot similarity heatmaps ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].imshow(sim_hidden, cmap='viridis')
    axs[0].set_title("Similarity: Hidden States")
    axs[1].imshow(sim_embed, cmap='viridis')
    axs[1].set_title("Similarity: Embeddings")

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    st.pyplot(fig)

    return r, p


