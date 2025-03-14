<template>
    <transition name="slide-fade" appear>
        <div>
            <h1 v-if="isAdd" class="mb-3">{{ $t("AI Target Deploy") }}</h1>
            <h1 v-else class="mb-3">
                <Uptime :stack="globalStack" :pill="true" /> {{ targetStackName }}
                <span v-if="$root.agentCount > 1" class="agent-name">
                    ({{ endpointDisplay }})
                </span>
            </h1>

            <div v-if="stack.isManagedByRedAiRange" class="mb-3">
                <div class="btn-group me-2" role="group">
                    <button v-if="isEditMode" class="btn btn-primary" :disabled="processing" @click="deployStack">
                        <font-awesome-icon icon="rocket" class="me-1" />
                        {{ $t("deployStack") }}
                    </button>
                    <button v-if="isEditMode" class="btn btn-normal" :disabled="processing" @click="saveStack">
                        <font-awesome-icon icon="save" class="me-1" />
                        {{ $t("saveStackDraft") }}
                    </button>
                    <button v-if="!isEditMode" class="btn btn-secondary" :disabled="processing" @click="enableEditMode">
                        <font-awesome-icon icon="pen" class="me-1" />
                        {{ $t("editStack") }}
                    </button>
                    <button v-if="!isEditMode && !active" class="btn btn-primary" :disabled="processing" @click="startStack">
                        <font-awesome-icon icon="play" class="me-1" />
                        {{ $t("startStack") }}
                    </button>
                    <button v-if="!isEditMode && active" class="btn btn-normal" :disabled="processing" @click="restartStack">
                        <font-awesome-icon icon="rotate" class="me-1" />
                        {{ $t("restartStack") }}
                    </button>
                    <button v-if="!isEditMode" class="btn btn-normal" :disabled="processing" @click="updateStack">
                        <font-awesome-icon icon="cloud-arrow-down" class="me-1" />
                        {{ $t("updateStack") }}
                    </button>
                    <button v-if="!isEditMode && active" class="btn btn-normal" :disabled="processing" @click="stopStack">
                        <font-awesome-icon icon="stop" class="me-1" />
                        {{ $t("stopStack") }}
                    </button>
                    <BDropdown right text="" variant="normal">
                        <BDropdownItem @click="downStack">
                            <font-awesome-icon icon="stop" class="me-1" />
                            {{ $t("downStack") }}
                        </BDropdownItem>
                    </BDropdown>
                </div>

                <button v-if="isEditMode && !isAdd" class="btn btn-normal" :disabled="processing" @click="discardStack">
                    {{ $t("discardStack") }}
                </button>
                <button v-if="!isEditMode" class="btn btn-danger" :disabled="processing" @click="showDeleteDialog = !showDeleteDialog">
                    <font-awesome-icon icon="trash" class="me-1" />
                    {{ $t("deleteStack") }}
                </button>
            </div>

            <!-- URLs -->
            <div v-if="urls.length > 0" class="mb-3">
                <a v-for="(url, index) in urls" :key="index" target="_blank" :href="url.url">
                    <span class="badge bg-secondary me-2">{{ url.display }}</span>
                </a>
            </div>

            <!-- Progress Terminal -->
            <transition name="slide-fade" appear>
                <Terminal
                    v-show="showProgressTerminal"
                    ref="progressTerminal"
                    class="mb-3 terminal"
                    :name="terminalName"
                    :endpoint="endpoint"
                    :rows="progressTerminalRows"
                    @has-data="showProgressTerminal = true; submitted = true;"
                ></Terminal>
            </transition>

            <div v-if="stack.isManagedByRedAiRange" class="row">
                <div class="col-lg-6">
                    <!-- General -->
                    <div v-if="isAdd">
                        <h4 class="mb-3">{{ $t("general") }}</h4>
                        <div class="shadow-box big-padding mb-3">
                            <!-- Stack Name -->
                            <div>
                                <label for="name" class="form-label">{{ $t("stackName") }}</label>
                                <input id="name" v-model="stack.name" type="text" class="form-control" required @blur="stackNameToLowercase">
                                <div class="form-text">{{ $t("Lowercase only") }}</div>
                            </div>

                            <!-- Endpoint -->
                            <div class="mt-3">
                                <label for="name" class="form-label">{{ $t("redairangeAgent") }}</label>
                                <select v-model="stack.endpoint" class="form-select">
                                    <option v-for="(agent, endpoint) in $root.agentList" :key="endpoint" :value="endpoint" :disabled="$root.agentStatusList[endpoint] != 'online'">
                                        ({{ $root.agentStatusList[endpoint] }}) {{ (endpoint) ? endpoint : $t("currentEndpoint") }}
                                    </option>
                                </select>
                            </div>
                        </div>
                    </div>

                    <!-- Containers -->
                    <h4 class="mb-3">{{ $tc("container", 2) }}</h4>

                    <div v-if="isEditMode" class="input-group mb-3">
                        <input
                            v-model="newContainerName"
                            :placeholder="$t(`New Container Name...`)"
                            class="form-control"
                            @keyup.enter="addContainer"
                        />
                        <button class="btn btn-primary" @click="addContainer">
                            {{ $t("addContainer") }}
                        </button>
                    </div>

                    <div ref="containerList">
                        <Container
                            v-for="(service, name) in jsonConfig.services"
                            :key="name"
                            :name="name"
                            :is-edit-mode="isEditMode"
                            :first="name === Object.keys(jsonConfig.services)[0]"
                            :status="serviceStatusList[name]"
                        />
                    </div>

                    <button v-if="false && isEditMode && jsonConfig.services && Object.keys(jsonConfig.services).length > 0" class="btn btn-normal mb-3" @click="addContainer">
                        {{ $t("addContainer") }}
                    </button>

                    <!-- General -->
                    <div v-if="isEditMode">
                        <h4 class="mb-3">{{ $t("extra") }}</h4>
                        <div class="shadow-box big-padding mb-3">
                            <!-- URLs -->
                            <div class="mb-4">
                                <label class="form-label">
                                    {{ $tc("url", 2) }}
                                </label>
                                <ArrayInput name="urls" :display-name="$t('url')" placeholder="https://" object-type="x-redairange" />
                            </div>
                        </div>
                    </div>

                    <!-- Combined Terminal Output -->
                    <div v-show="!isEditMode">
                        <h4 class="mb-3">{{ $t("terminal") }}</h4>
                        <Terminal
                            ref="combinedTerminal"
                            class="mb-3 terminal"
                            :name="combinedTerminalName"
                            :endpoint="endpoint"
                            :rows="combinedTerminalRows"
                            :cols="combinedTerminalCols"
                            style="height: 315px;"
                        ></Terminal>
                    </div>
                </div>
                <div class="col-lg-6">
                    <h4 class="mb-3">{{ stack.composeFileName }}</h4>

                    <!-- YAML editor -->
                    <div class="shadow-box mb-3 editor-box" :class="{'edit-mode' : isEditMode}">
                        <prism-editor
                            ref="editor"
                            v-model="stack.composeYAML"
                            class="yaml-editor"
                            :highlight="highlighterYAML"
                            line-numbers :readonly="!isEditMode"
                            @input="yamlCodeChange"
                            @focus="editorFocus = true"
                            @blur="editorFocus = false"
                        ></prism-editor>
                    </div>
                    <div v-if="isEditMode" class="mb-3">
                        {{ yamlError }}
                    </div>

                    <!-- ENV editor -->
                    <div v-if="isEditMode">
                        <h4 class="mb-3">.env</h4>
                        <div class="shadow-box mb-3 editor-box" :class="{'edit-mode' : isEditMode}">
                            <prism-editor
                                ref="editor"
                                v-model="stack.composeENV"
                                class="env-editor"
                                :highlight="highlighterENV"
                                line-numbers :readonly="!isEditMode"
                                @focus="editorFocus = true"
                                @blur="editorFocus = false"
                            ></prism-editor>
                        </div>
                    </div>

                    <div v-if="isEditMode">
                        <!-- Networks -->
                        <h4 class="mb-3">{{ $tc("network", 2) }}</h4>
                        <div class="shadow-box big-padding mb-3">
                            <NetworkInput />
                        </div>
                    </div>
                </div>
            </div>

            <div v-if="!stack.isManagedByRedAiRange && !processing">
                {{ $t("stackNotManagedByRedAiRangeMsg") }}
            </div>

            <!-- Delete Dialog -->
            <BModal v-model="showDeleteDialog" :okTitle="$t('deleteStack')" okVariant="danger" @ok="deleteDialog">
                {{ $t("deleteStackMsg") }}
            </BModal>
        </div>
    </transition>
</template>

<script>
import { highlight, languages } from "prismjs/components/prism-core";
import { PrismEditor } from "vue-prism-editor";
import "prismjs/components/prism-yaml";
import { parseDocument, Document } from "yaml";
import "prismjs/themes/prism-tomorrow.css";
import "vue-prism-editor/dist/prismeditor.min.css";
import { FontAwesomeIcon } from "@fortawesome/vue-fontawesome";
import {
    COMBINED_TERMINAL_COLS,
    COMBINED_TERMINAL_ROWS,
    copyYAMLComments,
    envsubstYAML,
    getCombinedTerminalName,
    getComposeTerminalName,
    PROGRESS_TERMINAL_ROWS,
    RUNNING,
} from "../../../common/util-common";
import { BModal } from "bootstrap-vue-next";
import NetworkInput from "../components/NetworkInput.vue";
import dotenv from "dotenv";

const template = `
version: '3.8'
services:
  target:
    stdin_open: true
    tty: true
    #user: root # Run container as root
    ports:
      - 11000:8888
    volumes:
      - .:/home/jovyan/work
    image: quay.io/jupyter/pytorch-notebook:latest
networks: {}
  `;
const envDefault = "# VARIABLE=value #comment # OPENAI_API_KEY= sk-123";

let yamlErrorTimeout = null;
let serviceStatusTimeout = null;
let prismjsSymbolDefinition = {
    symbol: {
        pattern: /(?<!\$)\$(\{[^{}]*\}|\w+)/,
    },
};

export default {
    components: {
        NetworkInput,
        FontAwesomeIcon,
        PrismEditor,
        BModal,
    },
    beforeRouteUpdate(to, from, next) {
        this.exitConfirm(next);
    },
    beforeRouteLeave(to, from, next) {
        this.exitConfirm(next);
    },
    yamlDoc: null,
    data() {
        return {
            editorFocus: false,
            jsonConfig: {},
            envsubstJSONConfig: {},
            yamlError: "",
            processing: true,
            showProgressTerminal: false,
            progressTerminalRows: PROGRESS_TERMINAL_ROWS,
            combinedTerminalRows: COMBINED_TERMINAL_ROWS,
            combinedTerminalCols: COMBINED_TERMINAL_COLS,
            stack: {},
            serviceStatusList: {},
            isEditMode: false,
            submitted: false,
            showDeleteDialog: false,
            newContainerName: "",
            stopServiceStatusTimeout: false,
        };
    },
    computed: {
        // Computed property: add stackname to  _ai_target
        targetStackName() {
            if (!this.stack.name) {
                return "";
            }
            return this.stack.name.endsWith("_ai_target")
                ? this.stack.name
                : this.stack.name + "_ai_target";
        },
        endpointDisplay() {
            return this.$root.endpointDisplayFunction(this.endpoint);
        },
        urls() {
            if (
                !this.envsubstJSONConfig["x-redairange"] ||
          !this.envsubstJSONConfig["x-redairange"].urls ||
          !Array.isArray(this.envsubstJSONConfig["x-redairange"].urls)
            ) {
                return [];
            }
            let urls = [];
            for (const url of this.envsubstJSONConfig["x-redairange"].urls) {
                let display;
                try {
                    let obj = new URL(url);
                    let pathname = obj.pathname;
                    if (pathname === "/") {
                        pathname = "";
                    }
                    display = obj.host + pathname + obj.search;
                } catch (e) {
                    display = url;
                }
                urls.push({
                    display,
                    url,
                });
            }
            return urls;
        },
        isAdd() {
            return this.$route.path === "/composetarget" && !this.submitted;
        },
        globalStack() {
            return this.$root.completeStackList[this.targetStackName + "_" + this.endpoint];
        },
        status() {
            return this.globalStack?.status;
        },
        active() {
            return this.status === RUNNING;
        },
        terminalName() {
            if (!this.stack.name) {
                return "";
            }
            return getComposeTerminalName(this.endpoint, this.targetStackName);
        },
        combinedTerminalName() {
            if (!this.stack.name) {
                return "";
            }
            return getCombinedTerminalName(this.endpoint, this.targetStackName);
        },
        networks() {
            return this.jsonConfig.networks;
        },
        endpoint() {
            return this.stack.endpoint || this.$route.params.endpoint || "";
        },
        url() {
            if (this.stack.endpoint) {
                return `/compose/${this.targetStackName}/${this.stack.endpoint}`;
            } else {
                return `/compose/${this.targetStackName}`;
            }
        },
    },
    watch: {
        "stack.composeYAML": {
            handler() {
                if (this.editorFocus) {
                    console.debug("yaml code changed");
                    this.yamlCodeChange();
                }
            },
            deep: true,
        },
        "stack.composeENV": {
            handler() {
                if (this.editorFocus) {
                    console.debug("env code changed");
                    this.yamlCodeChange();
                }
            },
            deep: true,
        },
        jsonConfig: {
            handler() {
                if (!this.editorFocus) {
                    console.debug("jsonConfig changed");
                    let doc = new Document(this.jsonConfig);
                    if (this.yamlDoc) {
                        copyYAMLComments(doc, this.yamlDoc);
                    }
                    this.stack.composeYAML = doc.toString();
                    this.yamlDoc = doc;
                }
            },
            deep: true,
        },
        $route(to, from) {},
    },
    mounted() {
        if (this.isAdd) {
            this.processing = false;
            this.isEditMode = true;
            let composeYAML;
            let composeENV;
            if (this.$root.composeTemplate) {
                composeYAML = this.$root.composeTemplate;
                this.$root.composeTemplate = "";
            } else {
                composeYAML = template;
            }
            if (this.$root.envTemplate) {
                composeENV = this.$root.envTemplate;
                this.$root.envTemplate = "";
            } else {
                composeENV = envDefault;
            }
            this.stack = {
                name: "",
                composeYAML,
                composeENV,
                isManagedByRedAiRange: true,
                endpoint: "",
            };
            this.yamlCodeChange();
        } else {
            this.stack.name = this.$route.params.stackName;
            this.loadStack();
        }
        this.requestServiceStatus();
    },
    methods: {
        startServiceStatusTimeout() {
            clearTimeout(serviceStatusTimeout);
            serviceStatusTimeout = setTimeout(async () => {
                this.requestServiceStatus();
            }, 5000);
        },
        requestServiceStatus() {
            this.$root.emitAgent(this.endpoint, "serviceStatusList", this.stack.name, (res) => {
                if (res.ok) {
                    this.serviceStatusList = res.serviceStatusList;
                }
                if (!this.stopServiceStatusTimeout) {
                    this.startServiceStatusTimeout();
                }
            });
        },
        exitConfirm(next) {
            if (this.isEditMode) {
                if (confirm("You are currently editing a stack. Are you sure you want to leave?")) {
                    this.exitAction();
                    next();
                } else {
                    next(false);
                }
            } else {
                this.exitAction();
                next();
            }
        },
        exitAction() {
            console.log("exitAction");
            this.stopServiceStatusTimeout = true;
            clearTimeout(serviceStatusTimeout);
            console.debug("leaveCombinedTerminal", this.endpoint, this.stack.name);
            this.$root.emitAgent(this.endpoint, "leaveCombinedTerminal", this.stack.name, () => {});
        },
        bindTerminal() {
            this.$refs.progressTerminal?.bind(this.endpoint, this.terminalName);
        },
        loadStack() {
            this.processing = true;
            this.$root.emitAgent(this.endpoint, "getStack", this.stack.name, (res) => {
                if (res.ok) {
                    this.stack = res.stack;
                    this.yamlCodeChange();
                    this.processing = false;
                    this.bindTerminal();
                } else {
                    this.$root.toastRes(res);
                }
            });
        },
        // make lowercase and add _ai_target to end
        stackNameToLowercase() {
            if (!this.stack.name) {
                return;
            }
            this.stack.name = this.stack.name.toLowerCase();
            if (!this.stack.name.endsWith("_ai_target")) {
                this.stack.name += "_ai_target";
            }
        },
        deployStack() {
            this.processing = true;
            if (!this.jsonConfig.services) {
                this.$root.toastError("No services found in compose.yaml");
                this.processing = false;
                return;
            }
            if (typeof this.jsonConfig.services !== "object") {
                this.$root.toastError("Services must be an object");
                this.processing = false;
                return;
            }
            let serviceNameList = Object.keys(this.jsonConfig.services);
            if (!this.stack.name && serviceNameList.length > 0) {
                let serviceName = serviceNameList[0];
                let service = this.jsonConfig.services[serviceName];
                if (service && service.container_name) {
                    this.stack.name = service.container_name;
                } else {
                    this.stack.name = serviceName;
                }
                if (!this.stack.name.endsWith("_ai_target")) {
                    this.stack.name += "_ai_target";
                }
            }
            this.bindTerminal();
            this.$root.emitAgent(
                this.stack.endpoint,
                "deployStack",
                this.targetStackName,
                this.stack.composeYAML,
                this.stack.composeENV,
                this.isAdd,
                (res) => {
                    this.processing = false;
                    this.$root.toastRes(res);
                    if (res.ok) {
                        this.isEditMode = false;
                        this.$router.push(this.url);
                    }
                }
            );
        },
        saveStack() {
            this.processing = true;
            this.$root.emitAgent(
                this.stack.endpoint,
                "saveStack",
                this.targetStackName,
                this.stack.composeYAML,
                this.stack.composeENV,
                this.isAdd,
                (res) => {
                    this.processing = false;
                    this.$root.toastRes(res);
                    if (res.ok) {
                        this.isEditMode = false;
                        this.$router.push(this.url);
                    }
                }
            );
        },
        startStack() {
            this.processing = true;
            this.$root.emitAgent(this.endpoint, "startStack", this.targetStackName, (res) => {
                this.processing = false;
                this.$root.toastRes(res);
            });
        },
        stopStack() {
            this.processing = true;
            this.$root.emitAgent(this.endpoint, "stopStack", this.targetStackName, (res) => {
                this.processing = false;
                this.$root.toastRes(res);
            });
        },
        downStack() {
            this.processing = true;
            this.$root.emitAgent(this.endpoint, "downStack", this.targetStackName, (res) => {
                this.processing = false;
                this.$root.toastRes(res);
            });
        },
        restartStack() {
            this.processing = true;
            this.$root.emitAgent(this.endpoint, "restartStack", this.targetStackName, (res) => {
                this.processing = false;
                this.$root.toastRes(res);
            });
        },
        updateStack() {
            this.processing = true;
            this.$root.emitAgent(this.endpoint, "updateStack", this.targetStackName, (res) => {
                this.processing = false;
                this.$root.toastRes(res);
            });
        },
        deleteDialog() {
            this.$root.emitAgent(this.endpoint, "deleteStack", this.targetStackName, (res) => {
                this.$root.toastRes(res);
                if (res.ok) {
                    this.$router.push("/");
                }
            });
        },
        discardStack() {
            this.loadStack();
            this.isEditMode = false;
        },
        highlighterYAML(code) {
            if (!languages.yaml_with_symbols) {
                languages.yaml_with_symbols = languages.insertBefore("yaml", "punctuation", {
                    symbol: prismjsSymbolDefinition["symbol"],
                });
            }
            return highlight(code, languages.yaml_with_symbols);
        },
        highlighterENV(code) {
            if (!languages.docker_env) {
                languages.docker_env = {
                    comment: {
                        pattern: /(^#| #).*$/m,
                        greedy: true,
                    },
                    keyword: {
                        pattern: /^\w*(?=[:=])/m,
                        greedy: true,
                    },
                    value: {
                        pattern: /(?<=[:=]).*?((?= #)|$)/m,
                        greedy: true,
                        inside: {
                            string: [
                                {
                                    pattern: /^ *'.*?(?<!\\)'/m,
                                },
                                {
                                    pattern: /^ *".*?(?<!\\)"|^.*$/m,
                                    inside: prismjsSymbolDefinition,
                                },
                            ],
                        },
                    },
                };
            }
            return highlight(code, languages.docker_env);
        },
        yamlToJSON(yaml) {
            let doc = parseDocument(yaml);
            if (doc.errors.length > 0) {
                throw doc.errors[0];
            }
            const config = doc.toJS() ?? {};
            if (!config.services) {
                config.services = {};
            }
            if (Array.isArray(config.services) || typeof config.services !== "object") {
                throw new Error("Services must be an object");
            }
            return { config,
                doc };
        },
        yamlCodeChange() {
            try {
                let { config, doc } = this.yamlToJSON(this.stack.composeYAML);
                this.yamlDoc = doc;
                this.jsonConfig = config;
                let env = dotenv.parse(this.stack.composeENV);
                let envYAML = envsubstYAML(this.stack.composeYAML, env);
                this.envsubstJSONConfig = this.yamlToJSON(envYAML).config;
                clearTimeout(yamlErrorTimeout);
                this.yamlError = "";
            } catch (e) {
                clearTimeout(yamlErrorTimeout);
                if (this.yamlError) {
                    this.yamlError = e.message;
                } else {
                    yamlErrorTimeout = setTimeout(() => {
                        this.yamlError = e.message;
                    }, 3000);
                }
            }
        },
        enableEditMode() {
            this.isEditMode = true;
        },
        checkYAML() {},
        addContainer() {
            this.checkYAML();
            if (this.jsonConfig.services[this.newContainerName]) {
                this.$root.toastError("Container name already exists");
                return;
            }
            if (!this.newContainerName) {
                this.$root.toastError("Container name cannot be empty");
                return;
            }
            this.jsonConfig.services[this.newContainerName] = {
                restart: "unless-stopped",
            };
            this.newContainerName = "";
            let element = this.$refs.containerList.lastElementChild;
            element.scrollIntoView({
                block: "start",
                behavior: "smooth",
            });
        },
    },
};
</script>

  <style scoped lang="scss">
  @import "../styles/vars.scss";

  .terminal {
    height: 200px;
  }

  .editor-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    &.edit-mode {
      background-color: #2c2f38 !important;
    }
  }

  .agent-name {
    font-size: 13px;
    color: $dark-font-color3;
  }
  </style>
