<template>
    <div class="recorder">
        <video ref="video" :src="blobUrl" controls autoplay poster="/redai.png"></video>
        <p>Status: {{ status }}</p>
        <div class="buttons">
            <button
                v-if="status === 'idle' || status === 'permission-requested' || status === 'error'"
                @click="startRecording"
            >
                Start Recording
            </button>
            <button v-if="status === 'recording' || status === 'paused'" @click="stopRecording">
                Stop Recording
            </button>
            <button
                v-if="status === 'recording' || status === 'paused'"
                @click="togglePauseResume"
            >
                {{ status === 'paused' ? 'Resume Recording' : 'Pause Recording' }}
            </button>
            <button v-if="status === 'stopped'" @click="resetRecording">
                Reset Recording
            </button>
        </div>
        <p v-if="error" class="error">Error: {{ error.message || error }}</p>
    </div>
</template>

<script setup>
import { ref } from "vue";

// States: "recording", "idle", "error", "stopped", "paused", "permission-requested"
const blobUrl = ref(null);
const blob = ref(null);
const error = ref(null);
const mediaRecorder = ref(null);
const status = ref("permission-requested");
const streams = ref({ audio: null,
    screen: null });

// İsteğe bağlı: Ses kaydı aktif olsun mu?
const audioEnabled = true;

// Ekran (ve varsa ses) akışını isteme ve MediaRecorder'ı oluşturma fonksiyonu
const requestMediaStream = async () => {
    try {
        // Ekran kaydı için izin iste
        const displayMedia = await navigator.mediaDevices.getDisplayMedia();
        let userMedia = null;
        if (audioEnabled) {
            userMedia = await navigator.mediaDevices.getUserMedia({ audio: true });
        }
        const displayTracks = displayMedia.getTracks();
        const userTracks = userMedia ? userMedia.getTracks() : [];
        const tracks = [ ...displayTracks, ...userTracks ];
        status.value = "idle";

        // Yeni bir MediaStream oluşturup MediaRecorder ile kayda başla
        const stream = new MediaStream(tracks);
        const recorder = new MediaRecorder(stream);
        recorder.ondataavailable = (event) => {
            const url = window.URL.createObjectURL(event.data);
            blobUrl.value = url;
            blob.value = event.data;
        };
        mediaRecorder.value = recorder;

        streams.value = {
            audio: userMedia ? userMedia.getTracks().find((t) => t.kind === "audio") : null,
            screen: displayMedia.getTracks().find((t) => t.kind === "video") || null
        };
        return recorder;
    } catch (e) {
        error.value = e;
        status.value = "error";
    }
};

// Kayıta başlama
const startRecording = async () => {
    let recorder = mediaRecorder.value;
    if (!recorder) {
        recorder = await requestMediaStream();
    }
    if (recorder) {
        recorder.start();
        status.value = "recording";
    }
};

// Kaydı durdurma
const stopRecording = () => {
    if (!mediaRecorder.value) {
        throw new Error("No media stream!");
    }
    mediaRecorder.value.stop();
    status.value = "stopped";
    mediaRecorder.value.stream.getTracks().forEach((track) => track.stop());
    mediaRecorder.value = null;
};

// Kaydı duraklatma
const pauseRecording = () => {
    if (!mediaRecorder.value) {
        throw new Error("No media stream!");
    }
    mediaRecorder.value.pause();
    status.value = "paused";
};

// Kayda devam ettirme
const resumeRecording = () => {
    if (!mediaRecorder.value) {
        throw new Error("No media stream!");
    }
    mediaRecorder.value.resume();
    status.value = "recording";
};

// Kaydı sıfırlama
const resetRecording = () => {
    blobUrl.value = null;
    error.value = null;
    mediaRecorder.value = null;
    status.value = "idle";
};

// Duraklatma / devam ettirme butonu için toggle fonksiyonu
const togglePauseResume = () => {
    if (status.value === "recording") {
        pauseRecording();
    } else if (status.value === "paused") {
        resumeRecording();
    }
};
</script>

  <style scoped>
  .recorder {
    max-width: 800px;
    margin: 0 auto;
    padding: 16px;
    text-align: center;
  }

  video {
    width: 100%;
    border-radius: 8px;
    margin-bottom: 16px;
  }

  .buttons button {
    margin: 0 8px;
    padding: 8px 16px;
    cursor: pointer;
  }

  .error {
    color: red;
    margin-top: 16px;
  }
  </style>
