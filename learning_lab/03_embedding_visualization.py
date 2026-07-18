"""
03_embedding_visualization.py
================================
CONCEPT: Dimensionality reduction (PCA, t-SNE) - literally SEEING why
embeddings work by projecting 512-dimensional vectors down to 2D.

WHAT THIS SCRIPT DOES
-----------------------
Decrypts every registered user's embedding, then:

1. PCA (Principal Component Analysis): finds the 2 directions of greatest
   VARIANCE in the 512-d embedding space and projects every point onto
   them. It's a linear, fast, deterministic technique - good for a quick
   sanity-check plot.

2. t-SNE (t-distributed Stochastic Neighbor Embedding): a non-linear
   technique that tries to preserve LOCAL neighborhood structure (points
   close in 512-d stay close in 2D) rather than global variance. Often
   produces visually tighter, more separated clusters than PCA for
   embedding data, at the cost of being non-deterministic-looking (distances
   between clusters aren't meaningful, only which points cluster together).

WHY THIS MATTERS FOR YOUR PROJECT
--------------------------------------
If ArcFace embeddings are doing their job, each registered person's
webcam captures (if you register the same person multiple times) should
visually cluster tightly and separately from other people's clusters. This
is the most intuitive possible demonstration that "embeddings encode
identity" - much more convincing in a viva than reciting the concept.

NOTE: with only ONE embedding per registered user (one photo each), you'll
see individual POINTS, not clusters. To see actual clustering, register the
same person 3-4 times under slightly different conditions (temporarily,
using different external_ids like "abhay_1", "abhay_2") purely for this
demo, then delete the duplicates afterward from the Manage Users page.

RUN
----
python learning_lab/03_embedding_visualization.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import database as db
import privacy

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("LEARNING LAB 03: Embedding Visualization (PCA + t-SNE)")
    print("=" * 70)

    users = db.get_all_users_with_embeddings()
    if len(users) < 2:
        print("\nNeed at least 2 registered users. Register a couple of people first.")
        return

    embeddings = []
    labels = []
    for u in users:
        embeddings.append(privacy.decrypt_embedding(u["embedding_enc"]))
        labels.append(u["name"])

    X = np.array(embeddings)
    print(f"\nLoaded {X.shape[0]} embeddings, each {X.shape[1]}-dimensional.")

    # ---- PCA ----
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    print(f"\nPCA: first 2 components explain {explained.sum()*100:.1f}% of total variance "
          f"(component 1: {explained[0]*100:.1f}%, component 2: {explained[1]*100:.1f}%)")

    plt.figure(figsize=(7, 6))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for i, name in enumerate(unique_labels):
        idx = [j for j, l in enumerate(labels) if l == name]
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=name, color=colors[i], s=80)
    plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    plt.title("PCA Projection of Face Embeddings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_embeddings.png"), dpi=150)
    plt.close()
    print("Saved learning_lab/output/pca_embeddings.png")

    # ---- t-SNE ----
    # t-SNE's perplexity parameter must be less than the number of samples;
    # we cap it automatically so this doesn't crash on small galleries.
    from sklearn.manifold import TSNE
    n_samples = X.shape[0]
    perplexity = max(2, min(30, n_samples - 1))

    if n_samples < 4:
        print("\nSkipping t-SNE: needs at least ~4 samples to produce a meaningful layout "
              "(you have {}). PCA above is still valid.".format(n_samples))
        return

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(7, 6))
    for i, name in enumerate(unique_labels):
        idx = [j for j, l in enumerate(labels) if l == name]
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=name, color=colors[i], s=80)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title(f"t-SNE Projection of Face Embeddings (perplexity={perplexity})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_embeddings.png"), dpi=150)
    plt.close()
    print("Saved learning_lab/output/tsne_embeddings.png")

    print("\nInterpretation reminder: in t-SNE, only relative CLUSTERING matters -")
    print("the distance BETWEEN clusters and absolute axis values are not meaningful.")


if __name__ == "__main__":
    main()
