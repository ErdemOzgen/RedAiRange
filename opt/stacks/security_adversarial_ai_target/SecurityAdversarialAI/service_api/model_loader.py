import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import hashlib
import h5py
import io
import tensorflow as tf



def load_key(file_path):
    try:
        with open(file_path, 'rb') as file:  # Read the key as bytes
            key = file.read().strip()  # Read the key and strip any extra whitespace
        if len(key) not in (16, 24, 32):
            raise ValueError("AESGCM key must be 128, 192, or 256 bits (16, 24, or 32 bytes).")
        return key
    except FileNotFoundError:
        print("The file was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def decrypt_model_in_memory(encrypted_model_path, key_path, model_hash):
    key = load_key(key_path)
    if key is None:
        raise ValueError("Failed to load encryption key")

    with open(encrypted_model_path, 'rb') as file:
        encrypted_data = file.read()

    # Calculate the hash of the encrypted model
    encrypted_hash = hashlib.sha256(encrypted_data).hexdigest()
    print(f"Encrypted model hash: {encrypted_hash} and expected model hash: {model_hash}")
    if encrypted_hash != model_hash:
        raise ValueError("Hash mismatch: The encrypted model's hash does not match the expected hash")
    print("Hash check passed")
    # Decrypt the model
    nonce = encrypted_data[:12]
    ciphertext = encrypted_data[12:]
    aesgcm = AESGCM(key)
    decrypted_data = aesgcm.decrypt(nonce, ciphertext, None)
    print("Decryption successful")
    return decrypted_data

def load_decrypted_model(encrypted_model_path, key_path, model_hash):
    decrypted_model_data = decrypt_model_in_memory(encrypted_model_path, key_path, model_hash)
    
    # Use h5py to read the model from the decrypted bytes
    with h5py.File(io.BytesIO(decrypted_model_data), 'r') as h5_file:
        model = tf.keras.models.load_model(h5_file)
    
    return model
