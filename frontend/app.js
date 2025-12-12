// Dark Mode
const toggle = document.getElementById("darkSwitch");
toggle.addEventListener("change", () => {
    document.body.classList.toggle("dark");
});

// Predict Function
async function predict() {
    let fileInput = document.getElementById("imageInput");
    if (fileInput.files.length === 0) {
        alert("Please select an image!");
        return;
    }

    let file = fileInput.files[0];

    // Image Preview
    let preview = document.getElementById("preview");
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";

    // Loader animation
    let loader = document.getElementById("loader");
    loader.style.display = "block";

    document.getElementById("result").innerHTML = "";
    document.getElementById("treatment").style.display = "none";

    let formData = new FormData();
    formData.append("file", file);

    let response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
    });

    loader.style.display = "none";

    let data = await response.json();

    if (data.status === "error") {
        document.getElementById("result").innerHTML = "❌ " + data.message;
        return;
    }

    document.getElementById("result").innerHTML =
        `🌿 Prediction: <b>${data.class}</b><br>🔥 Confidence: ${(data.confidence * 100).toFixed(2)}%`;

    document.getElementById("treatment").innerHTML = data.treatment;
    document.getElementById("treatment").style.display = "block";
}
