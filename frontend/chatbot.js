const chatBox = document.getElementById("chat-box");
const chatInput = document.getElementById("chat-input");
// const backendUrl = "https://test-openai-chat.onrender.com"; // Replace with your actual backend
const backendUrl = "http:localhost:8000"; // Replace with your actual backend



chatInput.addEventListener("keydown", async (e) => {
  if (e.key === "Enter") {
    const question = chatInput.value;
    chatBox.innerHTML += `<div><b>You:</b> ${question}</div>`;
    chatInput.value = "";

    const response = await fetch(`${backendUrl}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });

    const data = await response.json();
    chatBox.innerHTML += `<div><b>Bot:</b> ${data.answer}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
  }
});
