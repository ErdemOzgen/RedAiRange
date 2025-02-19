<template>
    <div class="container-fluid">
        <div class="row">
            <div v-if="!$root.isMobile" class="col-12 col-md-4 col-xl-3">
                <div class="row">
                    <div class="col-12 col-md-6 mb-3 d-flex">
                        <router-link
                            to="/manual"
                            class="btn btn-primary w-100 d-flex align-items-center justify-content-center"
                        >
                            <font-awesome-icon icon="book" class="text-white" />
                            <span class="text-white">{{ $t("Range Manual") }}</span>

                        </router-link>
                    </div>

                    <div class="col-12 col-md-6 mb-3 d-flex">
                        <router-link
                            to="/compose"
                            class="btn btn-primary w-100 d-flex align-items-center justify-content-center"
                        >
                        <font-awesome-icon icon="screwdriver-wrench" class="text-white" />
                        <span class="text-white">{{ $t("compose") }}</span>

                        </router-link>
                    </div>

                    <div class="col-12 col-md-6 mb-3 d-flex">
                        <router-link
                            to="/composearsenal"
                            class="btn btn-primary w-100 d-flex align-items-center justify-content-center"
                        >
                        <font-awesome-icon icon="person-rifle" class="text-white" />
                        <span class="text-white">{{ $t("Arsenal") }}</span>

                        </router-link>
                    </div>

                    <div class="col-12 col-md-6 mb-3 d-flex">
                        <router-link
                            to="/composetarget"
                            class="btn btn-primary w-100 d-flex align-items-center justify-content-center"
                        >
                        <font-awesome-icon icon="bullseye" class="text-white" />
                        <span class="text-white">{{ $t("Target Machines") }}</span>
                        </router-link>
                    </div>
                </div>

                <StackList :scrollbar="true" />
            </div>

            <div ref="container" class="col-12 col-md-8 col-xl-9 mb-3">
                <!-- Add :key to disable vue router re-use the same component -->
                <router-view :key="$route.fullPath" :calculatedHeight="height" />
            </div>
        </div>
    </div>
</template>

<script>

import StackList from "../components/StackList.vue";

export default {
    components: {
        StackList,
    },
    data() {
        return {
            height: 0
        };
    },
    mounted() {
        this.height = this.$refs.container.offsetHeight;
    },
};
</script>

<style lang="scss" scoped>
.container-fluid {
    width: 98%;
}
</style>
