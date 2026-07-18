"""
privacy.py
==========
MODULE 2: PRIVACY LAYER

WHY ENCRYPT EMBEDDINGS AT ALL?
-------------------------------
A face embedding is a 512-dimensional vector of floating point numbers.
It is NOT a picture, but research has shown embeddings can sometimes be
partially inverted back into a face-like image (a "model inversion
attack"). Because embeddings are biometric data, we treat them the same
way we would treat a password hash: encrypt at rest, so that even if the
database file is copied by an attacker, the raw vectors are unreadable
without the secret key.

HOW IT WORKS
------------
We use AES (Advanced Encryption Standard) in GCM mode via PyCryptodome:
  - AES-GCM is a modern "authenticated encryption" mode: it both encrypts
    the data AND lets us detect if it was tampered with (via a MAC tag).
  - A random 256-bit key is generated once and stored in
    database/secret.key (a plain binary file). NEVER commit this file to
    git - .gitignore already excludes it. Losing this file means all
    registered faces must re-enroll, because the embeddings become
    permanently unreadable (this is *correct* security behaviour, not a
    bug).
  - Each encryption call uses a fresh random 12-byte nonce so encrypting
    the same embedding twice produces different ciphertext.

We deliberately do NOT use homomorphic encryption (matching embeddings
directly in encrypted form) because it is computationally heavy and out
of scope for an MVP - see the project brief. Instead, embeddings are
decrypted in-memory only, at the moment they are needed for comparison,
and never written back to disk unencrypted.
"""

import os
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import numpy as np

from config import KEY_FILE_PATH

NONCE_SIZE = 12  # bytes, recommended size for AES-GCM


def _load_or_create_key() -> bytes:
    """Load the AES-256 key from disk, generating one on first run."""
    if os.path.exists(KEY_FILE_PATH):
        with open(KEY_FILE_PATH, "rb") as f:
            return f.read()
    key = get_random_bytes(32)  # 256-bit key
    with open(KEY_FILE_PATH, "wb") as f:
        f.write(key)
    return key


_KEY = _load_or_create_key()


def encrypt_embedding(embedding: np.ndarray) -> bytes:
    """
    Serialize a float32 numpy embedding vector to bytes and encrypt it.
    Output layout: [nonce (12 bytes)][auth tag (16 bytes)][ciphertext]
    """
    raw_bytes = embedding.astype(np.float32).tobytes()
    cipher = AES.new(_KEY, AES.MODE_GCM, nonce=get_random_bytes(NONCE_SIZE))
    ciphertext, tag = cipher.encrypt_and_digest(raw_bytes)
    return cipher.nonce + tag + ciphertext


def decrypt_embedding(blob: bytes) -> np.ndarray:
    """Reverse of encrypt_embedding(). Returns the original float32 vector."""
    nonce = blob[:NONCE_SIZE]
    tag = blob[NONCE_SIZE:NONCE_SIZE + 16]
    ciphertext = blob[NONCE_SIZE + 16:]
    cipher = AES.new(_KEY, AES.MODE_GCM, nonce=nonce)
    raw_bytes = cipher.decrypt_and_verify(ciphertext, tag)
    return np.frombuffer(raw_bytes, dtype=np.float32)
