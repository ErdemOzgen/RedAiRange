import secrets

def generate_secret_key():
    return secrets.token_hex(32)

def main():
    secret_key = generate_secret_key()
    print(f"Generated SECRET_KEY: {secret_key}")

    other_vars = {}
    while True:
        key = input("Enter environment variable key (or leave blank to finish): ")
        if not key:
            break
        value = input(f"Enter value for {key}: ")
        other_vars[key] = value

    print("\nEnvironment Variables:")
    print(f"SECRET_KEY={secret_key}")
    for key, value in other_vars.items():
        print(f"{key}={value}")

if __name__ == '__main__':
    main()
