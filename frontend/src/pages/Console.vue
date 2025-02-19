<template>
    <transition name="slide-fade" appear>
        <div>
            <h1 class="mb-3">Console</h1>

            <div>
                <p>
                    {{ $t("Example commands:") }}
                    <template v-for="(command, index) in allowedCommandList" :key="command">
                        <code>{{ command }}</code>

                        <!-- No comma at the end -->
                        <span v-if="index !== allowedCommandList.length - 1">, </span>
                    </template>
                </p>
                <p style="color: red;">
                    Be Aware: This is not fully functional terminal. Main purpose help you control simple jobs without connecting to the cyber range.
                </p>
            </div>

            <Terminal class="terminal" :rows="20" mode="mainTerminal" name="console" :endpoint="endpoint"></Terminal>
        </div>
    </transition>
</template>

<script>

import { allowedCommandList } from "../../../common/util-common";

export default {
    components: {
    },
    data() {
        return {
            allowedCommandList,
        };
    },
    computed: {
        endpoint() {
            return this.$route.params.endpoint || "";
        },
    },
    mounted() {

    },
    methods: {

    }
};
</script>

<style scoped lang="scss">
.terminal {
    height: 410px;
}
</style>
