<template>
    <div class="manual-page">
        <div class="image-container">
            <img src="/redai.png" alt="Redai" style="width: 750px; height: auto;" />
        </div>
        <h1>Welcome to Red Ai Range</h1>
        <p>This is the manual page content.</p>
    </div>
</template>

<script>
export default {
    name: "ManualPage",
    data() {
        return {
            // Add your data properties here
        };
    },
    mounted() {
        // Add your mounted logic here
    },
    methods: {
        // Add your methods here
    }
};
</script>

<style scoped>
.manual-page {
    padding: 20px;
}

.manual-page h1 {
    font-size: 24px;
    margin-bottom: 10px;
}

.manual-page p {
    font-size: 16px;
}
</style>