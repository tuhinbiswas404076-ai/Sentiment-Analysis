const input = document.getElementById('text-input');
const charCount = document.getElementById('char-count');
const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn = document.getElementById('clear-btn');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('error-message');
const resultCard = document.getElementById('result-card');
const resultSentiment = document.getElementById('result-sentiment');
const confidenceValue = document.getElementById('confidence-value');
const positiveBar = document.getElementById('positive-bar');
const neutralBar = document.getElementById('neutral-bar');
const negativeBar = document.getElementById('negative-bar');
const positiveProb = document.getElementById('positive-prob');
const neutralProb = document.getElementById('neutral-prob');
const negativeProb = document.getElementById('negative-prob');

function updateCharCount() {
  charCount.textContent = `${input.value.length} / 5000`;
}

function setLoading(isLoading) {
  loading.classList.toggle('hidden', !isLoading);
  analyzeBtn.disabled = isLoading;
  analyzeBtn.textContent = isLoading ? 'Analyzing...' : 'Analyze Sentiment';
}

function renderResult(payload) {
  resultCard.classList.remove('hidden');
  errorMessage.classList.add('hidden');
  resultSentiment.textContent = payload.sentiment.charAt(0).toUpperCase() + payload.sentiment.slice(1);
  confidenceValue.textContent = `${Math.round(payload.confidence * 100)}%`;

  const probs = payload.probabilities || {};
  const positive = Math.round((probs.positive || 0) * 100);
  const neutral = Math.round((probs.neutral || 0) * 100);
  const negative = Math.round((probs.negative || 0) * 100);

  positiveBar.style.width = `${positive}%`;
  neutralBar.style.width = `${neutral}%`;
  negativeBar.style.width = `${negative}%`;
  positiveProb.textContent = `${positive}%`;
  neutralProb.textContent = `${neutral}%`;
  negativeProb.textContent = `${negative}%`;
}

async function analyzeText() {
  const text = input.value.trim();
  if (!text) {
    errorMessage.textContent = 'Please enter some text.';
    errorMessage.classList.remove('hidden');
    return;
  }

  setLoading(true);
  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const payload = await response.json();
    if (!payload.success) {
      throw new Error(payload.message || 'Prediction failed.');
    }
    renderResult(payload);
  } catch (error) {
    errorMessage.textContent = error.message;
    errorMessage.classList.remove('hidden');
  } finally {
    setLoading(false);
  }
}

input.addEventListener('input', updateCharCount);
clearBtn.addEventListener('click', () => {
  input.value = '';
  updateCharCount();
  resultCard.classList.add('hidden');
  errorMessage.classList.add('hidden');
});
analyzeBtn.addEventListener('click', analyzeText);
document.querySelectorAll('.example-btn').forEach((button) => {
  button.addEventListener('click', () => {
    input.value = button.dataset.text;
    updateCharCount();
    analyzeText();
  });
});

updateCharCount();
