document.addEventListener("DOMContentLoaded", function () {
    const submitBtn = document.getElementById("submit-btn");
    const resultDiv = document.getElementById("result");

    submitBtn.addEventListener("click", async function () {
        resultDiv.innerHTML = "⏳ Checking...";
        resultDiv.style.color = "yellow";

        const inputText = document.getElementById("features").value.trim();
        const featureValues = inputText.split(/\s+/).map(parseFloat);

        if (featureValues.length !== 30 || featureValues.includes(NaN)) {
            resultDiv.innerHTML = "❌ Please enter exactly 30 valid numeric values!";
            resultDiv.style.color = "red";
            return;
        }

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ features: featureValues })
            });

            const data = await response.json();
            if (data.fraudulent) {
                resultDiv.innerHTML = "🚨 Fraudulent Transaction Detected!";
                resultDiv.style.color = "red";
            } else {
                resultDiv.innerHTML = "✅ Transaction is Legitimate.";
                resultDiv.style.color = "green";
            }
        } catch (error) {
            resultDiv.innerHTML = "⚠️ Error: Unable to process request.";
            resultDiv.style.color = "orange";
        }
    });
});
