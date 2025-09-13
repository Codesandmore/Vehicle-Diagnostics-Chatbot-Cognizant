/**
 * Client-Side Speech Recognition Module
 * Handles audio recording, upload, and integration with server-side Whisper
 */

class SpeechRecognition {
  constructor() {
    this.isRecording = false;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.stream = null;

    // UI elements (will be set by initializeUI)
    this.voiceButton = null;
    this.userInput = null;

    // Configuration
    this.config = {
      mimeType: "audio/webm;codecs=opus", // Fallback to 'audio/webm' if not supported
      audioBitsPerSecond: 128000,
      maxRecordingTime: 300000, // 5 minutes max
      uploadTimeout: 30000, // 30 seconds upload timeout
    };

    // Check browser support
    this.isSupported = this.checkBrowserSupport();
  }

  /**
   * Check if browser supports required APIs
   */
  checkBrowserSupport() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.warn("🎤 getUserMedia not supported");
      return false;
    }

    if (!window.MediaRecorder) {
      console.warn("🎤 MediaRecorder not supported");
      return false;
    }

    return true;
  }

  /**
   * Initialize UI elements and event listeners
   */
  initializeUI(voiceButtonId, userInputId) {
    this.voiceButton = document.getElementById(voiceButtonId);
    this.userInput = document.getElementById(userInputId);

    if (!this.voiceButton || !this.userInput) {
      console.error("🎤 Required UI elements not found");
      return false;
    }

    // Set up event listeners
    this.voiceButton.addEventListener("click", () => {
      this.toggleRecording();
    });

    // Update button appearance based on support
    if (!this.isSupported) {
      this.voiceButton.disabled = true;
      this.voiceButton.title =
        "Speech recognition not supported in this browser";
      this.voiceButton.style.opacity = "0.5";
      return false;
    }

    console.log("✅ Speech recognition UI initialized");
    return true;
  }

  /**
   * Toggle recording state
   */
  async toggleRecording() {
    if (this.isRecording) {
      await this.stopRecording();
    } else {
      await this.startRecording();
    }
  }

  /**
   * Start audio recording
   */
  async startRecording() {
    if (!this.isSupported) {
      this.showError("Speech recognition not supported in this browser");
      return;
    }

    try {
      console.log("🎤 Requesting microphone access...");

      // Request microphone access
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        },
      });

      // Determine the best supported MIME type
      let mimeType = this.config.mimeType;
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = "audio/webm";
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = ""; // Let browser choose
        }
      }

      // Create MediaRecorder
      this.mediaRecorder = new MediaRecorder(this.stream, {
        mimeType: mimeType,
        audioBitsPerSecond: this.config.audioBitsPerSecond,
      });

      // Reset audio chunks
      this.audioChunks = [];

      // Set up event handlers
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        this.processRecording();
      };

      this.mediaRecorder.onerror = (event) => {
        console.error("🎤 MediaRecorder error:", event.error);
        this.showError("Recording error occurred");
        this.resetRecording();
      };

      // Start recording
      this.mediaRecorder.start(1000); // Collect data every second
      this.isRecording = true;

      // Update UI
      this.updateUIForRecording(true);

      // Set maximum recording time
      setTimeout(() => {
        if (this.isRecording) {
          console.log("🎤 Maximum recording time reached, stopping...");
          this.stopRecording();
        }
      }, this.config.maxRecordingTime);

      console.log("🎤 Recording started");
    } catch (error) {
      console.error("🎤 Error starting recording:", error);

      if (error.name === "NotAllowedError") {
        this.showError(
          "Microphone access denied. Please allow microphone access and try again."
        );
      } else if (error.name === "NotFoundError") {
        this.showError(
          "No microphone found. Please connect a microphone and try again."
        );
      } else {
        this.showError("Error starting recording: " + error.message);
      }

      this.resetRecording();
    }
  }

  /**
   * Stop audio recording
   */
  async stopRecording() {
    if (!this.isRecording || !this.mediaRecorder) {
      return;
    }

    console.log("🎤 Stopping recording...");

    try {
      this.mediaRecorder.stop();
      this.isRecording = false;

      // Stop all tracks
      if (this.stream) {
        this.stream.getTracks().forEach((track) => track.stop());
        this.stream = null;
      }

      // Update UI
      this.updateUIForRecording(false);
    } catch (error) {
      console.error("🎤 Error stopping recording:", error);
      this.showError("Error stopping recording");
      this.resetRecording();
    }
  }

  /**
   * Process the recorded audio
   */
  async processRecording() {
    if (this.audioChunks.length === 0) {
      this.showError("No audio recorded");
      this.resetRecording();
      return;
    }

    try {
      console.log("🎤 Processing recorded audio...");

      // Create blob from audio chunks
      let mimeType = this.mediaRecorder.mimeType || "audio/webm";
      const audioBlob = new Blob(this.audioChunks, { type: mimeType });

      console.log(
        `🎤 Audio blob created: ${audioBlob.size} bytes, type: ${mimeType}`
      );

      // Show processing state
      this.showProcessing(true);

      // Upload and transcribe
      await this.uploadAndTranscribe(audioBlob);
    } catch (error) {
      console.error("🎤 Error processing recording:", error);
      this.showError("Error processing recorded audio");
    } finally {
      this.showProcessing(false);
      this.resetRecording();
    }
  }

  /**
   * Upload audio to server and get transcription
   */
  async uploadAndTranscribe(audioBlob) {
    try {
      console.log("🎤 Uploading audio for transcription...");

      // Prepare form data
      const formData = new FormData();
      formData.append("audio", audioBlob, "recording.webm");

      // Upload with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(
        () => controller.abort(),
        this.config.uploadTimeout
      );

      const response = await fetch("/transcribe_audio", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(
          `Server error: ${response.status} ${response.statusText}`
        );
      }

      const result = await response.json();

      if (result.success) {
        console.log(`✅ Transcription successful: "${result.text}"`);
        this.appendToInput(result.text);
        this.showSuccess("Speech converted to text successfully!");
      } else {
        console.error("🎤 Transcription failed:", result.error);
        this.showError(result.error || "Speech recognition failed");
      }
    } catch (error) {
      if (error.name === "AbortError") {
        this.showError("Upload timeout - please try again with shorter audio");
      } else {
        console.error("🎤 Upload error:", error);
        this.showError("Failed to process speech: " + error.message);
      }
    }
  }

  /**
   * Append transcribed text to input field
   */
  appendToInput(text) {
    if (!text || !this.userInput) return;

    const currentText = this.userInput.value.trim();
    const newText = currentText ? `${currentText} ${text}` : text;

    this.userInput.value = newText;

    // Trigger input event for any listeners
    this.userInput.dispatchEvent(new Event("input", { bubbles: true }));

    // Focus the input
    this.userInput.focus();

    // Set cursor to end
    this.userInput.setSelectionRange(newText.length, newText.length);
  }

  /**
   * Update UI during recording
   */
  updateUIForRecording(isRecording) {
    if (!this.voiceButton) return;

    if (isRecording) {
      this.voiceButton.innerHTML = "⏹️"; // Stop icon
      this.voiceButton.title = "Stop recording";
      this.voiceButton.classList.add("recording");
      this.voiceButton.style.background =
        "linear-gradient(90deg, #dc3545 0%, #c82333 100%)";
    } else {
      this.voiceButton.innerHTML = "🎤"; // Microphone icon
      this.voiceButton.title = "Start voice input";
      this.voiceButton.classList.remove("recording");
      this.voiceButton.style.background =
        "linear-gradient(90deg, #28a745 0%, #20c997 100%)";
    }
  }

  /**
   * Show processing state
   */
  showProcessing(isProcessing) {
    if (!this.voiceButton) return;

    if (isProcessing) {
      this.voiceButton.innerHTML = "⏳";
      this.voiceButton.title = "Processing speech...";
      this.voiceButton.disabled = true;
      this.voiceButton.style.background = "#6c757d";
    } else {
      this.voiceButton.disabled = false;
      this.updateUIForRecording(false);
    }
  }

  /**
   * Reset recording state
   */
  resetRecording() {
    this.isRecording = false;
    this.audioChunks = [];

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    if (this.mediaRecorder) {
      this.mediaRecorder = null;
    }

    this.updateUIForRecording(false);
  }

  /**
   * Show error message to user
   */
  showError(message) {
    console.error("🎤 Speech Recognition Error:", message);

    // You can customize this to show errors in your UI
    // For now, we'll use a simple alert
    if (typeof addMessage === "function") {
      addMessage(`🎤 Speech Error: ${message}`);
    } else {
      alert(`Speech Recognition Error: ${message}`);
    }
  }

  /**
   * Show success message to user
   */
  showSuccess(message) {
    console.log("✅ Speech Recognition Success:", message);

    // You can customize this to show success in your UI
    if (typeof addMessage === "function") {
      addMessage(`✅ ${message}`);
    }
  }

  /**
   * Get current status
   */
  getStatus() {
    return {
      isSupported: this.isSupported,
      isRecording: this.isRecording,
      hasStream: !!this.stream,
      hasMediaRecorder: !!this.mediaRecorder,
    };
  }
}

// Export for global use
window.SpeechRecognition = SpeechRecognition;
