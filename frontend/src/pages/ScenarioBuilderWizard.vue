<template>
    <transition name="slide-fade" appear>
        <div>
            <h1 class="mb-3">Scenario Builder Wizard</h1>
            <p class="text-muted mb-4">Select a preset, adjust ports/image, and generate a compose draft for your next scenario.</p>

            <div class="shadow-box big-padding mb-4">
                <h4 class="mb-3">1) Deploy Mode</h4>
                <select v-model="deployMode" class="form-select">
                    <option value="compose">Compose</option>
                    <option value="target">Target</option>
                    <option value="arsenal">Arsenal</option>
                </select>
            </div>

            <div class="shadow-box big-padding mb-4">
                <div class="d-flex align-items-center justify-content-between mb-3">
                    <h4 class="mb-0">2) Preset</h4>
                    <button class="btn btn-normal" :disabled="loadingPresets" @click="loadPresets">Refresh</button>
                </div>

                <div v-if="loadingPresets" class="text-muted">Loading presets...</div>

                <div v-else-if="presetList.length === 0" class="text-muted">
                    No presets available.
                </div>

                <div v-else class="row g-3">
                    <div
                        v-for="preset in presetList"
                        :key="preset.id"
                        class="col-12 col-lg-4"
                    >
                        <div class="preset-card" :class="{ selected: preset.id === selectedPresetId }" @click="selectPreset(preset)">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <h5 class="mb-0">{{ preset.name }}</h5>
                                <span class="badge bg-secondary">{{ preset.defaultDeployMode }}</span>
                            </div>
                            <p class="small text-muted mb-2">{{ preset.description }}</p>
                            <div class="small text-muted">Image: {{ preset.defaultImage }}</div>
                            <div class="small text-muted">Default Port: {{ preset.defaultHostPort }}:{{ preset.defaultContainerPort }}</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="shadow-box big-padding mb-4">
                <h4 class="mb-3">3) Runtime Settings</h4>

                <div class="row g-3">
                    <div class="col-12 col-md-6">
                        <label class="form-label">Host Port</label>
                        <input v-model.number="hostPort" type="number" min="1" max="65535" class="form-control">
                    </div>

                    <div class="col-12 col-md-6">
                        <label class="form-label">Container Port</label>
                        <input v-model.number="containerPort" type="number" min="1" max="65535" class="form-control">
                    </div>

                    <div class="col-12 col-md-6">
                        <label class="form-label">Custom Image (optional)</label>
                        <input v-model="customImage" type="text" class="form-control" placeholder="quay.io/jupyter/pytorch-notebook:latest">
                    </div>

                    <div class="col-12 col-md-6">
                        <label class="form-label">Service Name (optional)</label>
                        <input v-model="serviceName" type="text" class="form-control" placeholder="target">
                    </div>

                    <div class="col-12">
                        <label class="form-label">Command Override (optional)</label>
                        <input v-model="command" type="text" class="form-control" placeholder="tail -f /dev/null">
                    </div>

                    <div class="col-12">
                        <label class="form-label">Environment Variables (KEY=VALUE, one per line)</label>
                        <textarea v-model="envText" class="form-control" rows="6" placeholder="JUPYTER_ENABLE_LAB=yes"></textarea>
                    </div>
                </div>
            </div>

            <div class="d-flex gap-2 mb-4">
                <button class="btn btn-primary" :disabled="generating || !selectedPresetId" @click="generateTemplate">
                    <template v-if="generating">Generating...</template>
                    <template v-else>Generate and Open Editor</template>
                </button>
                <button class="btn btn-normal" @click="$router.push('/')">Cancel</button>
            </div>
        </div>
    </transition>
</template>

<script>
export default {
    data() {
        return {
            loadingPresets: false,
            generating: false,
            presetList: [],
            selectedPresetId: "",
            deployMode: "target",
            hostPort: 11000,
            containerPort: 8888,
            customImage: "",
            serviceName: "",
            command: "",
            envText: "",
        };
    },
    mounted() {
        this.loadPresets();
    },
    methods: {
        loadPresets() {
            this.loadingPresets = true;

            this.$root.getSocket().emit("scenarioBuilderPresetList", (res) => {
                this.loadingPresets = false;

                if (!res.ok) {
                    this.$root.toastRes(res);
                    return;
                }

                this.presetList = Array.isArray(res.presets) ? res.presets : [];

                if (!this.selectedPresetId && this.presetList.length > 0) {
                    this.selectPreset(this.presetList[0]);
                }
            });
        },

        selectPreset(preset) {
            this.selectedPresetId = preset.id;
            this.deployMode = preset.defaultDeployMode || "target";
            this.hostPort = preset.defaultHostPort;
            this.containerPort = preset.defaultContainerPort;
            this.customImage = "";
            this.serviceName = preset.defaultServiceName || "";
            this.command = preset.defaultCommand || "";

            const defaultEnv = preset.defaultEnv || {};
            this.envText = Object.keys(defaultEnv)
                .map((key) => `${key}=${defaultEnv[key]}`)
                .join("\n");
        },

        parseEnvText() {
            const result = {};
            const lines = this.envText.split("\n");

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed || trimmed.startsWith("#")) {
                    continue;
                }

                const equalIndex = trimmed.indexOf("=");
                if (equalIndex <= 0) {
                    continue;
                }

                const key = trimmed.slice(0, equalIndex).trim();
                const value = trimmed.slice(equalIndex + 1).trim();

                if (key.length > 0) {
                    result[key] = value;
                }
            }

            return result;
        },

        generateTemplate() {
            if (!this.selectedPresetId) {
                this.$root.toastError("Please select a preset.");
                return;
            }

            this.generating = true;

            const payload = {
                presetId: this.selectedPresetId,
                deployMode: this.deployMode,
                hostPort: this.hostPort,
                containerPort: this.containerPort,
                customImage: this.customImage,
                serviceName: this.serviceName,
                command: this.command,
                env: this.parseEnvText(),
            };

            this.$root.getSocket().emit("scenarioBuilderGenerate", payload, (res) => {
                this.generating = false;

                if (!res.ok) {
                    this.$root.toastRes(res);
                    return;
                }

                this.$root.composeTemplate = res.composeYAML;
                this.$root.envTemplate = res.composeENV || "";

                const routeMap = {
                    compose: "/compose",
                    target: "/composetarget",
                    arsenal: "/composearsenal",
                };

                this.$router.push(routeMap[this.deployMode] || "/compose");
            });
        },
    },
};
</script>

<style scoped lang="scss">
.preset-card {
    border: 1px solid #d9d9d9;
    border-radius: 10px;
    padding: 14px;
    cursor: pointer;
    transition: all 0.2s ease;
    height: 100%;
}

.preset-card:hover {
    border-color: #4090ff;
}

.preset-card.selected {
    border-color: #4090ff;
    box-shadow: 0 0 0 2px rgba(64, 144, 255, 0.2);
}
</style>
