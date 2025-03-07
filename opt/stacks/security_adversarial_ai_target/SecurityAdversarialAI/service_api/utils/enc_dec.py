import argparse
import os
from cryptography.hazmat.primitives import algorithms, modes
from cryptography.hazmat.backends import default_backend

def encrypt(data, key):
    nonce = os.urandom(12)
    cipher = algorithms.AES(key)
    mode = modes.GCM(nonce)
    encryptor = default_backend().create_symmetric_encryption_ctx(cipher, mode)
    encrypted_data = encryptor.update(data) + encryptor.finalize()
    return nonce + encrypted_data

def decrypt(encrypted_data, key):
    nonce = encrypted_data[:12]
    cipher = algorithms.AES(key)
    mode = modes.GCM(nonce)
    decryptor = default_backend().create_symmetric_decryption_ctx(cipher, mode)
    decrypted_data = decryptor.update(encrypted_data[12:]) + decryptor.finalize()
    return decrypted_data

def main():
    parser = argparse.ArgumentParser(description="AES-256 GCM Encryption/Decryption Tool")
    parser.add_argument("operation", choices=["-e", "-d"], help="Choose '-e' for encryption or '-d' for decryption.")
    parser.add_argument("data", help="Data to be encrypted or decrypted.")
    parser.add_argument("key", help="AES-256 encryption key (32 bytes).")

    args = parser.parse_args()

    if len(args.key) != 32:
        print("Error: The key must be 32 bytes long for AES-256.")
        return

    if args.operation == "-e":
        encrypted_result = encrypt(args.data.encode(), args.key.encode())
        print(f"Encrypted Data: {encrypted_result.hex()}")
    elif args.operation == "-d":
        decrypted_result = decrypt(bytes.fromhex(args.data), args.key.encode())
        print(f"Decrypted Data: {decrypted_result.decode()}")

if __name__ == "__main__":
    main()
