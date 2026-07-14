const form = document.querySelector("#prediction-form");
const ticket = document.querySelector("#ticket");
const clearButton = document.querySelector("#clear-ticket");
const submitButton = form.querySelector("button[type=submit]");
const characterCount = document.querySelector("#character-count");
const resultBox = document.querySelector("#result-box");
const resultStatus = document.querySelector(".result-status");
const resultPriority = document.querySelector("#result-priority");
const resultDetail = document.querySelector("#result-detail");
const probabilities = document.querySelector("#probabilities");
const probabilityList = document.querySelector("#probability-list");

function updateCharacterCount() {
  characterCount.textContent = `${ticket.value.length.toLocaleString()} characters`;
}

function setResult(state, status, title, detail) {
  resultBox.dataset.state = state;
  resultStatus.textContent = status;
  resultPriority.textContent = title;
  resultDetail.textContent = detail;
}

function renderProbabilities(values) {
  probabilityList.replaceChildren();
  Object.entries(values).forEach(([label, value]) => {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const definition = document.createElement("dd");
    term.textContent = label;
    definition.textContent = `${(value * 100).toFixed(1)}%`;
    row.append(term, definition);
    probabilityList.append(row);
  });
  probabilities.hidden = false;
}

async function loadMetrics() {
  try {
    const response = await fetch("/metrics");
    if (!response.ok) return;
    const metrics = await response.json();
    const macroF1 = metrics?.test?.macro_f1;
    const p95 = metrics?.optimization?.int8?.p95_ms;
    if (typeof macroF1 === "number") {
      document.querySelector("#macro-f1").textContent = macroF1.toFixed(3);
      document.querySelector("#evaluation-note").textContent = `Held-out test set · run ${metrics.run_id}`;
    }
    if (typeof p95 === "number") {
      document.querySelector("#latency").textContent = `${p95.toFixed(1)} ms`;
      document.querySelector("#runtime-note").textContent = "Measured with ONNX Runtime CPU inference.";
    }
  } catch (_) {
    // Metrics are optional until a Modal artifact is promoted.
  }
}

ticket.addEventListener("input", updateCharacterCount);
clearButton.addEventListener("click", () => {
  ticket.value = "";
  probabilities.hidden = true;
  setResult("idle", "Ready to classify", "Paste a ticket", "Your deployed model will return priority and confidence here.");
  updateCharacterCount();
  ticket.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = ticket.value.trim();
  if (!value) {
    ticket.focus();
    return;
  }
  submitButton.disabled = true;
  probabilities.hidden = true;
  setResult("loading", "Classifying ticket", "Working…", "Running the promoted ONNX model.");
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticket: value }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The service could not classify this ticket.");
    setResult(
      "success",
      "Priority",
      payload.priority[0].toUpperCase() + payload.priority.slice(1),
      `${(payload.confidence * 100).toFixed(1)}% confidence from the promoted model.`,
    );
    renderProbabilities(payload.probabilities);
  } catch (error) {
    setResult("error", "Classification unavailable", "Not ready", error.message);
  } finally {
    submitButton.disabled = false;
  }
});

updateCharacterCount();
loadMetrics();
