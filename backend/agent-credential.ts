import { createCipheriv, createDecipheriv, randomBytes } from "node:crypto";
import { Settings } from "./settings";

const ALGORITHM = "aes-256-gcm";
const ENCRYPTION_PREFIX = "enc:v1";
const AGENT_CREDENTIAL_KEY_SETTING = "agentCredentialKeyV1";

let cachedCredentialKey: Buffer | null = null;

function parseCredentialKey(rawKey: string): Buffer {
    const trimmed = rawKey.trim();

    // 64-hex format
    if (/^[0-9a-fA-F]{64}$/.test(trimmed)) {
        return Buffer.from(trimmed, "hex");
    }

    const fromBase64 = Buffer.from(trimmed, "base64");
    if (fromBase64.length !== 32) {
        throw new Error("Invalid REDAIRANGE_AGENT_CREDENTIALS_KEY length. Use 32-byte base64 or 64-char hex.");
    }
    return fromBase64;
}

async function getCredentialKey(): Promise<Buffer> {
    if (cachedCredentialKey) {
        return cachedCredentialKey;
    }

    const envKey = process.env.REDAIRANGE_AGENT_CREDENTIALS_KEY;
    if (typeof envKey === "string" && envKey.trim().length > 0) {
        cachedCredentialKey = parseCredentialKey(envKey);
        return cachedCredentialKey;
    }

    let storedKey = await Settings.get(AGENT_CREDENTIAL_KEY_SETTING);
    if (typeof storedKey !== "string" || storedKey.trim().length === 0) {
        storedKey = randomBytes(32).toString("base64");
        await Settings.set(AGENT_CREDENTIAL_KEY_SETTING, storedKey, "security");
    }

    cachedCredentialKey = parseCredentialKey(storedKey);
    return cachedCredentialKey;
}

export function isEncryptedAgentCredential(value: string): boolean {
    return value.startsWith(`${ENCRYPTION_PREFIX}:`);
}

export async function encryptAgentCredential(plainText: string): Promise<string> {
    const key = await getCredentialKey();
    const iv = randomBytes(12);
    const cipher = createCipheriv(ALGORITHM, key, iv);

    const encrypted = Buffer.concat([
        cipher.update(plainText, "utf8"),
        cipher.final(),
    ]);

    const authTag = cipher.getAuthTag();

    return `${ENCRYPTION_PREFIX}:${iv.toString("base64")}:${authTag.toString("base64")}:${encrypted.toString("base64")}`;
}

export async function decryptAgentCredential(encryptedValue: string): Promise<string> {
    if (!isEncryptedAgentCredential(encryptedValue)) {
        return encryptedValue;
    }

    const parts = encryptedValue.split(":");
    if (parts.length !== 5) {
        throw new Error("Invalid encrypted agent credential format.");
    }

    const iv = Buffer.from(parts[2], "base64");
    const authTag = Buffer.from(parts[3], "base64");
    const cipherText = Buffer.from(parts[4], "base64");

    const key = await getCredentialKey();
    const decipher = createDecipheriv(ALGORITHM, key, iv);
    decipher.setAuthTag(authTag);

    const decrypted = Buffer.concat([
        decipher.update(cipherText),
        decipher.final(),
    ]);

    return decrypted.toString("utf8");
}
