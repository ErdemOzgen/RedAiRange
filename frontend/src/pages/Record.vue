<template>
  <div class="recorder">
    <h2>Screen Recorder</h2>

    <video
      ref="videoEl"
      :src="blobUrl || undefined"
      controls
      playsinline
      preload="metadata"
      poster="/redai.png"
    ></video>

    <p>Status: {{ statusLabel }}</p>
    <p v-if="durationLabel" class="duration">Duration: {{ durationLabel }}</p>
    <p v-if="warning" class="warning">Warning: {{ warning }}</p>
    <p v-if="error" class="error">Error: {{ error }}</p>

    <div class="buttons">
      <button v-if="canStart" @click="startRecording">Start Recording</button>

      <button v-if="canStop" @click="stopRecording">Stop Recording</button>

      <button v-if="canPauseOrResume" @click="togglePauseResume">
        {{ status === "paused" ? "Resume Recording" : "Pause Recording" }}
      </button>

      <button v-if="canCancel" @click="cancelRecording">Cancel Recording</button>

      <button v-if="canDownload" @click="downloadRecording">Download Video</button>

      <button v-if="canReset" @click="resetRecording">New Recording</button>
    </div>
  </div>
</template>

<script setup>
import { computed, onUnmounted, ref } from "vue";
import { onBeforeRouteLeave } from "vue-router";

const audioEnabled = true;
const blobUrl = ref("");
const recordingBlob = ref(null);
const error = ref("");
const warning = ref("");
const status = ref("idle");
const mediaRecorder = ref(null);
const displayStream = ref(null);
const micStream = ref(null);
const mergedStream = ref(null);
const chunks = ref([]);
const keepRecordingResult = ref(true);
const recordingStart = ref(0);
const elapsedMs = ref(0);
const videoEl = ref(null);

let durationTimer = null;

const preferredMimeTypes = [
  "video/webm;codecs=vp9,opus",
  "video/webm;codecs=vp8,opus",
  "video/webm",
];

const statusLabel = computed(() => {
  if (status.value === "permission-requested") {
    return "permission-requested";
  }
  return status.value;
});

const durationLabel = computed(() => {
  if (status.value === "idle" && elapsedMs.value === 0) {
    return "";
  }

  const totalSec = Math.floor(elapsedMs.value / 1000);
  const minutes = String(Math.floor(totalSec / 60)).padStart(2, "0");
  const seconds = String(totalSec % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
});

const canStart = computed(() => ["idle", "stopped", "error"].includes(status.value));
const canStop = computed(() => ["recording", "paused"].includes(status.value));
const canPauseOrResume = computed(() => ["recording", "paused"].includes(status.value));
const canCancel = computed(() => ["recording", "paused", "permission-requested"].includes(status.value));
const canDownload = computed(() => Boolean(recordingBlob.value));
const canReset = computed(() => Boolean(recordingBlob.value) || status.value === "error" || status.value === "stopped");

const clearDurationTimer = () => {
  if (durationTimer) {
    window.clearInterval(durationTimer);
    durationTimer = null;
  }
};

const startDurationTimer = () => {
  clearDurationTimer();
  recordingStart.value = Date.now() - elapsedMs.value;
  durationTimer = window.setInterval(() => {
    elapsedMs.value = Date.now() - recordingStart.value;
  }, 250);
};

const stopDurationTimer = () => {
  clearDurationTimer();
};

const revokeBlobUrl = () => {
  if (blobUrl.value) {
    window.URL.revokeObjectURL(blobUrl.value);
    blobUrl.value = "";
  }
};

const stopStream = (stream) => {
  if (!stream) {
    return;
  }

  stream.getTracks().forEach((track) => track.stop());
};

const stopAllStreams = () => {
  stopStream(displayStream.value);
  stopStream(micStream.value);
  stopStream(mergedStream.value);

  displayStream.value = null;
  micStream.value = null;
  mergedStream.value = null;
};

const detachRecorderHandlers = () => {
  const recorder = mediaRecorder.value;
  if (!recorder) {
    return;
  }

  recorder.ondataavailable = null;
  recorder.onstop = null;
  recorder.onerror = null;
};

const pickMimeType = () => {
  return preferredMimeTypes.find((type) => MediaRecorder.isTypeSupported(type)) || "video/webm";
};

const requestMediaStream = async () => {
  if (!navigator.mediaDevices?.getDisplayMedia) {
    throw new Error("Screen recording is not supported in this browser or context (HTTPS required).");
  }

  status.value = "permission-requested";
  warning.value = "";

  const screen = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: audioEnabled,
  });

  let mic = null;

  if (audioEnabled) {
    try {
      mic = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
    } catch (audioErr) {
      warning.value = "Microphone not found/allowed. Continuing without microphone audio.";
      console.warn("Microphone unavailable. Continuing with available audio tracks:", audioErr);
    }
  }

  const videoTrack = screen.getVideoTracks()[0];
  const screenAudioTracks = screen.getAudioTracks();
  const micAudioTracks = mic ? mic.getAudioTracks() : [];
  const selectedAudioTracks = micAudioTracks.length > 0 ? micAudioTracks : screenAudioTracks;
  const selectedTracks = [videoTrack, ...selectedAudioTracks].filter(Boolean);

  if (selectedTracks.length === 0) {
    throw new Error("No media tracks available for recording.");
  }

  const stream = new MediaStream(selectedTracks);
  const mimeType = pickMimeType();
  const recorder = new MediaRecorder(stream, { mimeType });

  displayStream.value = screen;
  micStream.value = mic;
  mergedStream.value = stream;
  mediaRecorder.value = recorder;

  screen.getVideoTracks()[0]?.addEventListener("ended", () => {
    if (status.value === "recording" || status.value === "paused") {
      stopRecording();
    }
  });

  recorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      chunks.value.push(event.data);
    }
  };

  recorder.onerror = (event) => {
    const recorderError = event.error instanceof Error ? event.error.message : "Recording failed.";
    error.value = recorderError;
    status.value = "error";
    stopDurationTimer();
    stopAllStreams();
  };

  recorder.onstop = () => {
    stopDurationTimer();

    if (!keepRecordingResult.value || chunks.value.length === 0) {
      chunks.value = [];
      recordingBlob.value = null;
      if (status.value !== "error") {
        status.value = "idle";
      }
      return;
    }

    revokeBlobUrl();

    const resultBlob = new Blob(chunks.value, {
      type: recorder.mimeType || "video/webm",
    });

    chunks.value = [];
    recordingBlob.value = resultBlob;
    blobUrl.value = window.URL.createObjectURL(resultBlob);
    status.value = "stopped";
  };
};

const startRecording = async () => {
  if (status.value === "recording" || status.value === "paused") {
    return;
  }

  error.value = "";
  warning.value = "";
  elapsedMs.value = 0;
  keepRecordingResult.value = true;
  chunks.value = [];

  revokeBlobUrl();
  recordingBlob.value = null;

  try {
    await requestMediaStream();

    if (!mediaRecorder.value) {
      throw new Error("Recorder initialization failed.");
    }

    mediaRecorder.value.start(250);
    status.value = "recording";
    startDurationTimer();
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
    status.value = "error";
    stopDurationTimer();
    stopAllStreams();
  }
};

const stopRecording = () => {
  const recorder = mediaRecorder.value;
  if (!recorder) {
    return;
  }

  keepRecordingResult.value = true;
  status.value = "stopped";
  stopDurationTimer();

  if (recorder.state !== "inactive") {
    recorder.stop();
  }

  stopAllStreams();
};

const cancelRecording = () => {
  const recorder = mediaRecorder.value;

  keepRecordingResult.value = false;
  stopDurationTimer();
  elapsedMs.value = 0;
  chunks.value = [];

  if (recorder && recorder.state !== "inactive") {
    recorder.stop();
  }

  stopAllStreams();
  status.value = "idle";
};

const pauseRecording = () => {
  const recorder = mediaRecorder.value;
  if (!recorder || recorder.state !== "recording") {
    return;
  }

  recorder.pause();
  stopDurationTimer();
  status.value = "paused";
};

const resumeRecording = () => {
  const recorder = mediaRecorder.value;
  if (!recorder || recorder.state !== "paused") {
    return;
  }

  recorder.resume();
  startDurationTimer();
  status.value = "recording";
};

const togglePauseResume = () => {
  if (status.value === "recording") {
    pauseRecording();
  } else if (status.value === "paused") {
    resumeRecording();
  }
};

const downloadRecording = () => {
  if (!recordingBlob.value) {
    return;
  }

  const downloadUrl = blobUrl.value || window.URL.createObjectURL(recordingBlob.value);
  const link = document.createElement("a");
  const timestamp = new Date().toISOString().replace(/[.:]/g, "-");

  link.href = downloadUrl;
  link.download = `redairange-recording-${timestamp}.webm`;
  document.body.appendChild(link);
  link.click();
  link.remove();

  if (!blobUrl.value) {
    window.URL.revokeObjectURL(downloadUrl);
  }
};

const resetRecording = () => {
  cancelRecording();
  revokeBlobUrl();
  recordingBlob.value = null;
  error.value = "";
  warning.value = "";
  status.value = "idle";
};

const disposeOnLeave = () => {
  detachRecorderHandlers();

  const recorder = mediaRecorder.value;
  if (recorder && recorder.state !== "inactive") {
    try {
      recorder.stop();
    } catch (e) {
      console.warn("Recorder stop on leave failed:", e);
    }
  }

  mediaRecorder.value = null;
  stopDurationTimer();
  stopAllStreams();
  revokeBlobUrl();
  recordingBlob.value = null;
  chunks.value = [];
  elapsedMs.value = 0;
  status.value = "idle";
  error.value = "";
  warning.value = "";
};

onBeforeRouteLeave(() => {
  disposeOnLeave();
});

onUnmounted(() => {
  disposeOnLeave();
});
</script>

<style scoped>
.recorder {
  max-width: 920px;
  margin: 0 auto;
  padding: 16px;
  text-align: center;
}

video {
  width: 100%;
  min-height: 320px;
  border-radius: 8px;
  margin-bottom: 16px;
  background: #111;
}

.buttons {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
}

.buttons button {
  padding: 8px 16px;
  cursor: pointer;
}

.duration {
  color: #1f4e79;
  font-weight: 600;
}

.error {
  color: #bf1d1d;
  margin-top: 16px;
}

.warning {
  color: #a87000;
  margin-top: 8px;
}
</style>
  