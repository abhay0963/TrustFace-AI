"""
01_embeddings_and_similarity.py
=================================
CONCEPT: What is a face embedding, and how does cosine similarity actually
compare two of them?

WHAT THIS SCRIPT DOES
-----------------------
1. Pulls every registered user from the database and DECRYPTS their stored
   512-dimensional embedding (in-memory only, never written back to disk -
   same privacy boundary the live app uses).
2. Prints the first 10 numbers of one embedding so you can literally SEE
   what "a face, as a vector" looks like (a lot of small floats -
   demystifying: it's just numbers, not a photo).
3. Computes the FULL pairwise cosine similarity matrix between every
   registered user, by hand (not hidden inside a library call), so you can
   see exactly what the formula does:

       cosine_similarity(a, b) = (a . b) / (||a|| * ||b||)

4. Highlights: same-person similarity should be high (near the diagonal is
   always 1.0 - comparing someone to themselves), different-person
   similarity should be noticeably lower.

RUN
----
python learning_lab/01_embeddings_and_similarity.py
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import database as db
import privacy


def manual_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Written out longhand (not calling face_engine.cosine_similarity) so every step is visible."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)   # sqrt(sum of squares) - the vector's length
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def main():
    print("=" * 70)
    print("LEARNING LAB 01: Embeddings & Cosine Similarity")
    print("=" * 70)

    users = db.get_all_users_with_embeddings()
    if len(users) < 2:
        print("\nNeed at least 2 registered users for this demo.")
        print("Go register a couple of people at http://127.0.0.1:8000/register, then re-run this.")
        return

    print(f"\nFound {len(users)} registered users. Decrypting embeddings in-memory...\n")

    decrypted = []
    for u in users:
        emb = privacy.decrypt_embedding(u["embedding_enc"])
        decrypted.append({"name": u["name"], "external_id": u["external_id"], "embedding": emb})

    sample = decrypted[0]
    print(f"Embedding for '{sample['name']}' has shape {sample['embedding'].shape} "
          f"(a {sample['embedding'].shape[0]}-dimensional vector).")
    print(f"First 10 values: {np.round(sample['embedding'][:10], 4)}")
    print("Notice: these are just floating point numbers. There is no way to look at\n"
          "this array and 'see' a face - that's the whole point of the privacy design.\n")

    print("-" * 70)
    print("PAIRWISE COSINE SIMILARITY MATRIX")
    print("-" * 70)
    names = [d["name"] for d in decrypted]
    col_width = max(10, max(len(n) for n in names) + 2)

    header = " " * col_width + "".join(n.ljust(col_width) for n in names)
    print(header)
    for i, row_user in enumerate(decrypted):
        row = [row_user["name"].ljust(col_width)]
        for j, col_user in enumerate(decrypted):
            sim = manual_cosine_similarity(row_user["embedding"], col_user["embedding"])
            row.append(f"{sim:.4f}".ljust(col_width))
        print("".join(row))

    print("\nHow to read this:")
    print("  - The diagonal is always 1.0000 (a person compared to themselves).")
    print("  - Off-diagonal values close to 1.0 mean two DIFFERENT people whose")
    print("    embeddings ended up unexpectedly close - worth investigating if")
    print("    that similarity is anywhere near config.FACE_MATCH_THRESHOLD.")
    print("  - Off-diagonal values well below the threshold confirm ArcFace is")
    print("    doing its job: separating different identities in vector space.")


if __name__ == "__main__":
    main()
