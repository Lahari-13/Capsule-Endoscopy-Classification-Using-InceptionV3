function analyze() {
  let file = document.getElementById("image").files[0];
  let formData = new FormData();
  formData.append("image", file);

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById("result").innerHTML =
      `Prediction: ${data.prediction}<br>Confidence: ${data.confidence}%`;
  });
}
