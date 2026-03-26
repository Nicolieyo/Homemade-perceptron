const grid = document.getElementById("grid");
const sampleCount = document.getElementById("sampleCount");
const statusText = document.getElementById("status");
const predictionText = document.getElementById("prediction");

const clearBtn = document.getElementById("clearBtn");
const saveLBtn = document.getElementById("saveLBtn");
const saveTBtn = document.getElementById("saveTBtn");
const trainBtn = document.getElementById("trainBtn");
const predictBtn = document.getElementById("predictBtn");

const cells = [];
const dataset = [];
const gridSize = 16;

// creating actual 4x4 grid
for (let i = 0; i < gridSize; i++) {
  const cell = document.createElement("div");
  cell.classList.add("cell");

  cell.addEventListener("click", () => {
    cell.classList.toggle("active");
  });

  grid.appendChild(cell);
  cells.push(cell);
}

// changes grid to #'s
function getInputVector() {
  return cells.map(cell =>
    cell.classList.contains("active") ? 1 : 0
  );
}
function clearGrid() {
  cells.forEach(cell => cell.classList.remove("active"));
  predictionText.textContent = "Prediction: None";
}

// were saving here
function saveSample(label) {
  const input = getInputVector();
  dataset.push({ input, label });

  sampleCount.textContent = `Samples: ${dataset.length}`;
  statusText.textContent = `Saved sample as ${label === -1 ? "L" : "T"}`;

  clearGrid();
}

let weights = new Array(gridSize).fill(0);
let bias = 0;
let trained = false;

function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function activation(value) {
  return value >= 0 ? 1 : -1;
}

// some training
function trainPerceptron(epochs = 25, learningRate = 1) {
  weights = new Array(gridSize).fill(0);
  bias = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    let errors = 0;

    for (const sample of dataset) {
      const x = sample.input;
      const y = sample.label;

      const guess = activation(dotProduct(x, weights) + bias);

      if (guess !== y) {
        for (let i = 0; i < gridSize; i++) {
          weights[i] += learningRate * y * x[i];
        }
        bias += learningRate * y;
        errors++;
      }
    }

    if (errors === 0) break;
  }

  trained = true;
  statusText.textContent = "Training complete!";
}

// prediction on drawing
function predictSample() {
  if (!trained) {
    predictionText.textContent = "Train first!";
    return;
  }

  const input = getInputVector();
  const result = activation(dotProduct(input, weights) + bias);

  predictionText.textContent = `Prediction: ${result === -1 ? "L" : "T"}`;
}

// what the btns do
clearBtn.addEventListener("click", clearGrid);
saveLBtn.addEventListener("click", () => saveSample(-1));
saveTBtn.addEventListener("click", () => saveSample(1));

trainBtn.addEventListener("click", () => {
  if (dataset.length < 100) {
    statusText.textContent = "Need at least 100 samples!";
    return;
  }

  trainPerceptron();
});

predictBtn.addEventListener("click", predictSample);