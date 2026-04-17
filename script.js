const classNames = ["buildings", "forest", "glacier", "mountain", "sea", "street"];
let session = null;

// Change these paths only if your actual filenames are different
const sampleImages = {
  buildings: [
    "./samples/buildings/buildings1.jpg",
    "./samples/buildings/buildings2.jpg",
    "./samples/buildings/buildings3.jpg"
  ],
  forest: [
    "./samples/forest/forest1.jpg",
    "./samples/forest/forest2.jpg",
    "./samples/forest/forest3.jpg"
  ],
  glacier: [
    "./samples/glacier/glacier1.jpg",
    "./samples/glacier/glacier2.jpg",
    "./samples/glacier/glacier3.jpg"
  ],
  mountain: [
    "./samples/mountain/mountain1.jpg",
    "./samples/mountain/mountain2.jpg",
    "./samples/mountain/mountain3.jpg"
  ],
  sea: [
    "./samples/sea/sea1.jpg",
    "./samples/sea/sea2.jpg",
    "./samples/sea/sea3.jpg"
  ],
  street: [
    "./samples/street/street1.jpg",
    "./samples/street/street2.jpg",
    "./samples/street/street3.jpg"
  ]
};

async function loadModel() {
  const result = document.getElementById("result");

  try {
    result.innerText = "Loading model...";

    const modelResponse = await fetch("./intel_cnn.onnx");
    if (!modelResponse.ok) {
      throw new Error(`Could not fetch intel_cnn.onnx: ${modelResponse.status}`);
    }
    const modelBuffer = await modelResponse.arrayBuffer();
    console.log("Loaded .onnx bytes:", modelBuffer.byteLength);

    const dataResponse = await fetch("./intel_cnn.onnx.data");
    if (!dataResponse.ok) {
      throw new Error(`Could not fetch intel_cnn.onnx.data: ${dataResponse.status}`);
    }
    const dataBuffer = await dataResponse.arrayBuffer();
    console.log("Loaded .onnx.data bytes:", dataBuffer.byteLength);

    session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ["wasm"],
      externalData: [
        {
          path: "intel_cnn.onnx.data",
          data: new Uint8Array(dataBuffer)
        }
      ]
    });

    console.log("Model loaded successfully");
    console.log("Inputs:", session.inputNames);
    console.log("Outputs:", session.outputNames);

    result.innerText = "Model loaded successfully. Select an image and click Predict.";
  } catch (error) {
    console.error("Error loading model:", error);
    result.innerText = "Failed to load model: " + error.message;
  }
}

function loadSampleImages() {
  const category = document.getElementById("categorySelect").value;
  const gallery = document.getElementById("sampleGallery");
  gallery.innerHTML = "";

  if (!sampleImages[category] || sampleImages[category].length === 0) {
    gallery.innerHTML = "<p>No sample images found for this category.</p>";
    return;
  }

  sampleImages[category].forEach((imgPath) => {
    const img = document.createElement("img");
    img.src = imgPath;
    img.alt = category;

    img.onerror = () => {
      console.error("Image failed to load:", imgPath);
      img.style.display = "none";
    };

    img.onclick = () => {
      document.querySelectorAll(".gallery img").forEach(el => el.classList.remove("selected"));
      img.classList.add("selected");

      const preview = document.getElementById("preview");
      preview.src = imgPath;
    };

    gallery.appendChild(img);
  });
}

function preprocessImage(imgElement) {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgElement, 0, 0, 128, 128);

  const imageData = ctx.getImageData(0, 0, 128, 128).data;
  const floatData = new Float32Array(1 * 3 * 128 * 128);

  for (let i = 0; i < 128 * 128; i++) {
    const r = imageData[i * 4] / 255.0;
    const g = imageData[i * 4 + 1] / 255.0;
    const b = imageData[i * 4 + 2] / 255.0;

    floatData[i] = r;
    floatData[128 * 128 + i] = g;
    floatData[2 * 128 * 128 + i] = b;
    }
  return new ort.Tensor("float32", floatData, [1, 3, 128, 128]);
  }

function softmax(arr) {
  const maxVal = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

async function predictImage() {
  const result = document.getElementById("result");
  const preview = document.getElementById("preview");

  if (!session) {
    result.innerText = "Model is not loaded yet.";
    return;
  }

  if (!preview.src) {
    result.innerText = "Please select or upload an image first.";
    return;
  }

  try {
    result.innerText = "Running prediction...";

    const inputTensor = preprocessImage(preview);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    const outputs = await session.run({ [inputName]: inputTensor });
    const scores = Array.from(outputs[outputName].data);
    const probs = softmax(scores);

    let maxIndex = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIndex]) {
        maxIndex = i;
      }
    }

    result.innerHTML =
      `Prediction: <strong>${classNames[maxIndex]}</strong><br>` +
      `Confidence: ${(probs[maxIndex] * 100).toFixed(2)}%`;
  } catch (error) {
    console.error("Prediction error:", error);
    result.innerText = "Prediction failed: " + error.message;
  }
}

function clearSelection() {
  document.getElementById("preview").src = "";
  document.getElementById("imageUpload").value = "";
  document.getElementById("result").innerText = "Selection cleared.";
  document.querySelectorAll(".gallery img").forEach(el => el.classList.remove("selected"));
}

document.getElementById("loadSamplesBtn").addEventListener("click", loadSampleImages);
document.getElementById("predictBtn").addEventListener("click", predictImage);
document.getElementById("clearBtn").addEventListener("click", clearSelection);

document.getElementById("imageUpload").addEventListener("change", function(event) {
  const file = event.target.files[0];
  if (!file) return;

  const preview = document.getElementById("preview");
  preview.src = URL.createObjectURL(file);

  document.querySelectorAll(".gallery img").forEach(el => el.classList.remove("selected"));
});

loadSampleImages();
loadModel();
