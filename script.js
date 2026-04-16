const classNames = ["buildings", "forest", "glacier", "mountain", "sea", "street"];
let session;

async function loadModel() {
  session = await ort.InferenceSession.create("intel_cnn.onnx");
  console.log("Model loaded");
}

document.getElementById("imageUpload").addEventListener("change", function(event) {
  const file = event.target.files[0];
  if (file) {
    const img = document.getElementById("preview");
    img.src = URL.createObjectURL(file);
  }
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

async function predictImage() {
  const img = document.getElementById("preview");

  if (!img.src) {
    document.getElementById("result").innerText = "Please upload an image first.";
    return;
  }

  const inputTensor = preprocessImage(img);
  const outputs = await session.run({ input: inputTensor });

  const scores = outputs.output.data;

  let maxIndex = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[maxIndex]) {
      maxIndex = i;
    }
  }

  document.getElementById("result").innerText =
    "Prediction: " + classNames[maxIndex];
}

loadModel();
