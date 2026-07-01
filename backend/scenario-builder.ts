import yaml from "yaml";
import { ValidationError } from "./util-server";

export type ScenarioDeployMode = "compose" | "target" | "arsenal";

export interface ScenarioPreset {
    id: string;
    name: string;
    description: string;
    tags: string[];
    defaultDeployMode: ScenarioDeployMode;
    defaultImage: string;
    defaultServiceName: string;
    defaultHostPort: number;
    defaultContainerPort: number;
    defaultEnv: Record<string, string>;
    defaultCommand?: string;
    defaultUrlPath?: string;
}

export interface ScenarioBuilderRequest {
    presetId: string;
    deployMode: ScenarioDeployMode;
    hostPort: number;
    containerPort: number;
    customImage?: string;
    serviceName?: string;
    command?: string;
    env?: Record<string, string>;
}

export interface ScenarioBuilderResult {
    composeYAML: string;
    composeENV: string;
}

const PRESET_LIST: ScenarioPreset[] = [
    {
        id: "prompt-injection-lab",
        name: "Prompt Injection Lab",
        description: "Jupyter tabanli bir hedef ile prompt injection denemeleri icin temel ortam.",
        tags: [ "llm", "prompt-injection", "jupyter" ],
        defaultDeployMode: "target",
        defaultImage: "quay.io/jupyter/pytorch-notebook:latest",
        defaultServiceName: "target",
        defaultHostPort: 11000,
        defaultContainerPort: 8888,
        defaultEnv: {
            JUPYTER_ENABLE_LAB: "yes",
        },
        defaultUrlPath: "/",
    },
    {
        id: "model-extraction-toolkit",
        name: "Model Extraction Toolkit",
        description: "Arsenal tarafinda model extraction test araclari icin hazir konteyner.",
        tags: [ "model-extraction", "arsenal", "python" ],
        defaultDeployMode: "arsenal",
        defaultImage: "python:3.12-slim",
        defaultServiceName: "arsenal",
        defaultHostPort: 10104,
        defaultContainerPort: 8080,
        defaultEnv: {
            PYTHONUNBUFFERED: "1",
        },
        defaultCommand: "tail -f /dev/null",
        defaultUrlPath: "/",
    },
    {
        id: "rag-security-assessment",
        name: "RAG Security Assessment",
        description: "RAG saldiri yuzeyleri icin API tabanli test ortami.",
        tags: [ "rag", "retrieval", "api" ],
        defaultDeployMode: "compose",
        defaultImage: "python:3.12-slim",
        defaultServiceName: "rag_target",
        defaultHostPort: 12000,
        defaultContainerPort: 8000,
        defaultEnv: {
            APP_ENV: "dev",
            LOG_LEVEL: "info",
        },
        defaultCommand: "python -m http.server 8000",
        defaultUrlPath: "/",
    },
];

export function getScenarioPresetList() : ScenarioPreset[] {
    return PRESET_LIST;
}

function normalizePort(value: unknown, fieldName: string, fallback: number): number {
    if (value === null || value === undefined || value === "") {
        return fallback;
    }

    const num = Number(value);
    if (!Number.isInteger(num) || num < 1 || num > 65535) {
        throw new ValidationError(`${fieldName} must be an integer between 1 and 65535`);
    }

    return num;
}

function normalizeDeployMode(value: unknown, fallback: ScenarioDeployMode): ScenarioDeployMode {
    if (typeof value !== "string" || value.length === 0) {
        return fallback;
    }

    if (value === "compose" || value === "target" || value === "arsenal") {
        return value;
    }

    throw new ValidationError("deployMode must be one of: compose, target, arsenal");
}

function normalizeServiceName(value: unknown, fallback: string): string {
    if (typeof value !== "string" || value.trim().length === 0) {
        return fallback;
    }

    const normalized = value.trim().toLowerCase().replace(/[^a-z0-9_-]/g, "-");
    if (normalized.length === 0) {
        throw new ValidationError("serviceName is invalid");
    }

    return normalized;
}

function normalizeEnv(value: unknown, fallback: Record<string, string>): Record<string, string> {
    if (!value || typeof value !== "object") {
        return { ...fallback };
    }

    const result: Record<string, string> = { ...fallback };

    for (const [ key, rawValue ] of Object.entries(value as Record<string, unknown>)) {
        if (!key || typeof key !== "string") {
            continue;
        }
        if (typeof rawValue === "string") {
            result[key] = rawValue;
        } else if (typeof rawValue === "number" || typeof rawValue === "boolean") {
            result[key] = String(rawValue);
        }
    }

    return result;
}

function envToString(env: Record<string, string>): string {
    const keys = Object.keys(env);
    if (keys.length === 0) {
        return "# VARIABLE=value";
    }

    return keys.map((key) => `${key}=${env[key]}`).join("\n");
}

export function generateScenarioBuilderTemplate(requestData: unknown): ScenarioBuilderResult {
    if (!requestData || typeof requestData !== "object") {
        throw new ValidationError("Request must be an object");
    }

    const payload = requestData as Partial<ScenarioBuilderRequest>;

    if (typeof payload.presetId !== "string" || payload.presetId.trim().length === 0) {
        throw new ValidationError("presetId is required");
    }

    const preset = PRESET_LIST.find((item) => item.id === payload.presetId);
    if (!preset) {
        throw new ValidationError("Unknown presetId");
    }

    const deployMode = normalizeDeployMode(payload.deployMode, preset.defaultDeployMode);
    const hostPort = normalizePort(payload.hostPort, "hostPort", preset.defaultHostPort);
    const containerPort = normalizePort(payload.containerPort, "containerPort", preset.defaultContainerPort);
    const image = (typeof payload.customImage === "string" && payload.customImage.trim().length > 0)
        ? payload.customImage.trim()
        : preset.defaultImage;
    const serviceName = normalizeServiceName(payload.serviceName, preset.defaultServiceName);
    const command = (typeof payload.command === "string" && payload.command.trim().length > 0)
        ? payload.command.trim()
        : preset.defaultCommand;
    const env = normalizeEnv(payload.env, preset.defaultEnv);

    const composeObject: Record<string, unknown> = {
        services: {
            [serviceName]: {
                image,
                restart: "unless-stopped",
                stdin_open: true,
                tty: true,
                ports: [ `${hostPort}:${containerPort}` ],
                volumes: [ ".:/workspace" ],
                environment: env,
            },
        },
        networks: {},
        "x-redairange": {
            scenarioBuilder: {
                presetId: preset.id,
                deployMode,
            },
            urls: [ `http://localhost:${hostPort}${preset.defaultUrlPath || "/"}` ],
        },
    };

    const service = (composeObject.services as Record<string, Record<string, unknown>>)[serviceName];
    if (command) {
        service.command = command;
    }

    return {
        composeYAML: yaml.stringify(composeObject),
        composeENV: envToString(env),
    };
}
