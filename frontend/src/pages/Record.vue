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
  import { ref, onUnmounted } from "vue";
  
  // States: "recording", "idle", "error", "stopped", "paused", "permission-requested"
  const blobUrl = ref(null);
  const blob = ref(null);
  const error = ref(null);
  const mediaRecorder = ref(null);
  const status = ref("permission-requested");
  const streams = ref({ audio: null, screen: null });
  
  // Optional: enable audio recording
  const audioEnabled = true;
  
  // Request media streams and initialize MediaRecorder
  const requestMediaStream = async () => {
    try {
      const displayMedia = await navigator.mediaDevices.getDisplayMedia();
      let userMedia = null;
      if (audioEnabled) {
        userMedia = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
      const displayTracks = displayMedia.getTracks();
      const userTracks = userMedia ? userMedia.getTracks() : [];
      const tracks = [...displayTracks, ...userTracks];
      status.value = "idle";
  
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
  
  // Start recording function
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
  
  // Stop recording function
  const stopRecording = () => {
    if (!mediaRecorder.value) {
      throw new Error("No media stream!");
    }
    mediaRecorder.value.stop();
    status.value = "stopped";
    mediaRecorder.value.stream.getTracks().forEach((track) => track.stop());
    mediaRecorder.value = null;
  };
  
  // Pause recording
  const pauseRecording = () => {
    if (!mediaRecorder.value) {
      throw new Error("No media stream!");
    }
    mediaRecorder.value.pause();
    status.value = "paused";
  };
  
  // Resume recording
  const resumeRecording = () => {
    if (!mediaRecorder.value) {
      throw new Error("No media stream!");
    }
    mediaRecorder.value.resume();
    status.value = "recording";
  };
  
  // Reset recording state
  const resetRecording = () => {
    blobUrl.value = null;
    error.value = null;
    mediaRecorder.value = null;
    status.value = "idle";
  };
  
  // Toggle between pause and resume
  const togglePauseResume = () => {
    if (status.value === "recording") {
      pauseRecording();
    } else if (status.value === "paused") {
      resumeRecording();
    }
  };
  
  // Component cleanup: stop recording when component unmounts
  onUnmounted(() => {
    if (mediaRecorder.value && (status.value === "recording" || status.value === "paused")) {
      stopRecording();
    }
  });
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
  