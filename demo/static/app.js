// BlandVoice-Core Interactive Dashboard Controller

document.addEventListener("DOMContentLoaded", () => {
  // Tab Switching
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      document.getElementById(targetId).classList.add("active");
    });
  });

  // Preset Buttons
  const presetButtons = document.querySelectorAll(".preset-btn");
  const ttsText = document.getElementById("tts-text");
  presetButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      ttsText.value = btn.getAttribute("data-prompt");
    });
  });

  // Pitch Contour Canvas Visualizer
  const canvas = document.getElementById("canvas-pitch");
  const ctx = canvas.getContext("2d");

  function drawPitchContour(f0Data) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Background Grid
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1;
    for (let y = 20; y < canvas.height; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    if (!f0Data || f0Data.length === 0) return;

    const minF0 = 120.0;
    const maxF0 = 260.0;

    ctx.strokeStyle = "#8b5cf6";
    ctx.lineWidth = 3;
    ctx.beginPath();

    for (let i = 0; i < f0Data.length; i++) {
      const x = (i / (f0Data.length - 1)) * canvas.width;
      const normalizedF0 = (f0Data[i] - minF0) / (maxF0 - minF0);
      const y = canvas.height - (normalizedF0 * (canvas.height - 20) + 10);

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw baseline guideline
    ctx.strokeStyle = "rgba(59, 130, 246, 0.4)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Draw initial sample pitch contour
  const sampleF0 = [];
  for (let i = 0; i < 60; i++) {
    const t = i / 60;
    // Rising question curve
    const rise = t > 0.7 ? Math.pow((t - 0.7) / 0.3, 1.8) * 55.0 : 0;
    sampleF0.push(180 - 15 * t + rise + 6 * Math.sin(i * 0.4));
  }
  drawPitchContour(sampleF0);

  // Synthesize Button Click
  const btnSynthesize = document.getElementById("btn-synthesize");
  const audioPlayer = document.getElementById("tts-audio-player");
  const statusIndicator = document.getElementById("tts-status");

  btnSynthesize.addEventListener("click", async () => {
    const text = ttsText.value.trim();
    const emotion = document.getElementById("tts-emotion").value;
    const chunkFrames = parseInt(document.getElementById("tts-chunk").value);

    statusIndicator.innerText = "Synthesizing...";
    statusIndicator.style.color = "#3b82f6";

    try {
      const resp = await fetch("/api/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, emotion, chunk_frames: chunkFrames })
      });

      if (!resp.ok) {
        throw new Error(`Server returned HTTP ${resp.status}`);
      }

      const data = await resp.json();
      if (data.status === "ok") {
        document.getElementById("stat-ttfa").innerText = `${data.latency_ms.toFixed(1)} ms`;
        document.getElementById("stat-rtf").innerText = `${data.rtf.toFixed(2)}x`;
        document.getElementById("stat-duration").innerText = `${data.duration_sec.toFixed(2)} s`;
        
        const isQuestion = data.parsed_info.is_question;
        const inflEl = document.getElementById("stat-inflection");
        if (isQuestion) {
          inflEl.innerText = "Rising Terminal (+50 Hz)";
          inflEl.className = "telemetry-value text-accent";
        } else {
          inflEl.innerText = "Falling Declination (-25 Hz)";
          inflEl.className = "telemetry-value text-success";
        }

        drawPitchContour(data.f0_contour);

        // Load WAV into audio player
        audioPlayer.src = `data:audio/wav;base64,${data.audio_base64}`;
        audioPlayer.play();

        statusIndicator.innerText = "Streaming Complete";
        statusIndicator.style.color = "#10b981";
      } else {
        throw new Error(data.detail || "Synthesis failed");
      }
    } catch (err) {
      console.error(err);
      statusIndicator.innerText = "Service Busy (Retry)";
      statusIndicator.style.color = "#ef4444";
    }
  });

  // Telephony Simulation Slider
  const lossSlider = document.getElementById("sim-loss-slider");
  const lossValueLabel = document.getElementById("sim-loss-value");
  lossSlider.addEventListener("input", () => {
    lossValueLabel.innerText = `${lossSlider.value}% Burst Loss`;
  });

  // Run Telephony Simulation
  const btnRunTelephony = document.getElementById("btn-run-telephony");
  btnRunTelephony.addEventListener("click", async () => {
    btnRunTelephony.innerText = "Simulating Telephony Channel...";
    try {
      const resp = await fetch("/api/telephony_simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          loss_rate: parseFloat(lossSlider.value) / 100.0,
          standard: document.getElementById("sim-codec-select").value,
          enable_plc: document.getElementById("sim-plc-checkbox").checked
        })
      });

      const data = await resp.json();
      if (data.status === "ok") {
        document.getElementById("mcd-dropped").innerText = `${data.mcd_dropped.toFixed(2)} dB`;
        document.getElementById("sdr-dropped").innerText = `${data.sdr_dropped.toFixed(1)} dB`;
        document.getElementById("mcd-plc").innerText = `${data.mcd_plc.toFixed(2)} dB`;
        document.getElementById("sdr-plc").innerText = `${data.sdr_plc.toFixed(1)} dB (+${(data.sdr_plc - data.sdr_dropped).toFixed(1)} dB recovery)`;

        document.getElementById("audio-dropped").src = `data:audio/wav;base64,${data.audio_dropped_base64}`;
        document.getElementById("audio-plc").src = `data:audio/wav;base64,${data.audio_plc_base64}`;
      }
    } catch (e) {
      console.error(e);
    } finally {
      btnRunTelephony.innerText = "Run Telephony Simulation";
    }
  });
});
