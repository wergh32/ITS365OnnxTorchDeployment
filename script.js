const classNames = ["buildings", "forest", "glacier", "mountain", "sea", "street"];
let session = null;
let selectedImageElement = null;

// List sample images you placed in your repo
const sampleImages = {
  buildings: [
    "samples/buildings/buildings1.jpg",
    "samples/buildings/buildings2.jpg",
    "samples/buildings/buildings3.jpg"
  ],
  forest: [
    "samples/forest/forest1.jpg",
    "samples/forest/forest2.jpg",
    "samples/forest/forest3.jpg"
  ],
  glacier: [
    "samples/glacier/glacier1.jpg",
    "samples/glacier/glacier2.jpg",
    "samples/glacier/glacier3.jpg"
  ],
  mountain: [
    "samples/mountain/mountain1.jpg",
    "samples/mountain/mountain2.jpg",
    "samples/mountain/mountain3.jpg"
  ],
  sea: [
    "samples/sea/sea1.jpg",
    "samples/sea/sea2.jpg",
    "samples/sea/sea3.jpg"
  ],
  street: [
    "samples/street/street1.jpg",
    "samples/street/street2.jpg",
    "samples/street/street3.jpg"
  ]
};

async function loadModel() {
  try {
    session = await ort.InferenceSession.create("intel_cnn.onnx");
    console.log("Model loaded successfully");
  } catch (error) {
    console.error("Error loading model:", error);
    document.getElementById("result").innerText = "Failed to load ONNX model.";
  }
}

function loadSampleImages() {
  const category = document.getElementById("categorySelect").value;
  const gallery = document.getElementById("sampleGallery");
  gallery.innerHTML = "";

  sampleImages[category].forEach((imgPath) => {
    const img = document.createElement("img");
    img.src = imgPath;
    img.alt = category;

    img.addEventListener("click", () => {
      document.querySelectorAll(".gallery img").forEach(el => el.classList.remove("selected"));
      img.classList.add("selected");

      const preview = document.getElementById("preview");
      preview.src = imgPath;
      selectedImageElement = preview;
    });

    gallery.appendChild(img);
  });
}

document.getElementById("loadSamplesBtn").addEventListener("click", loadSampleImages);

document.getElementById("imageUpload").addEventListener("change", function(event) {
  const file = event.target.files[0];
  if (!file) return;

  const preview = document.getElementById("preview");
  preview.src = URL.createObjectURL(file);
  selectedImageElement = preview;

  document.querySelectorAll(".gallery img").forEach(el => el.classList.remove("selected"));
});

function preprocessImage(imgElement) {
  const canvas = document.createElement("canvas");
  canvas.width = 150;
  canvas.height = 150;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgElement, 0, 0, 150, 150);

  const imageData = ctx.getImageData(0, 0, 150, 150).data;
  const floatData = new Float32Array(1 * 3 * 150 * 150);

  for (let i = 0; i < 150 * 150; i++) {
    const r = imageData[i * 4] / 255.0;
    const g = imageData[i * 4 + 1] / 255.0;
    const b = imageData[i * 4 + 2] / 255.0;

    floatData[i] = r;
    floatData[150 * 150 + i] = g;
    floatData[2 * 150 * 150 + i] = b;
  }

  return new ort.Tensor("float32", floatData, [1, 3, 150, 150]);
}

function softmax(arr) {
  const maxVal = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

async function predictImage() {
  if (!session) {
    document.getElementById("result").innerText = "Model is still loading.";
    return;
  }

  const preview = document.getElementById("preview");
  if (!preview.src) {
    document.getElementById("result").innerText = "Please select or upload an image first.";
    return;
  }

  try {
    const inputTensor = preprocessImage(preview);
    const outputs = await session.run({ input: inputTensor });

    const scores = Array.from(outputs.output.data);
    const probs = softmax(scores);

    let maxIndex = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIndex]) {
        maxIndex = i;
      }
    }

    document.getElementById("result").innerHTML =
      `Prediction: <strong>${classNames[maxIndex]}</strong><br>
       Confidence: ${(probs[maxIndex] * 100).toFixed(2)}%`;
  } catch (error) {
    console.error("Prediction error:", error);
    document.getElementById("result").innerText = "Prediction failed.";
  }
}

document.getElementById("predictBtn").addEventListener("click", predictImage);

loadModel();
loadSampleImages();
