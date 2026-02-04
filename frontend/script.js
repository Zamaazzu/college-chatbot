function sendMessage() {
    const msg = document.getElementById("message").value;

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: msg })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText =
            "Detected intent: " + data.intent + "\n" + data.response;
    });
}
