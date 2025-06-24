const chatBox = document.getElementById("chat-box");
const chatInput = document.getElementById("chat-input");

chatInput.addEventListener("keydown", async (e) => {
  if (e.key === "Enter") {
    const question = chatInput.value;
    chatBox.innerHTML += `<div><b>You:</b> ${question}</div>`;
    chatInput.value = "";

    const response = await fetch("https://your-render-backend-url.onrender.com/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });

    const data = await response.json();
    chatBox.innerHTML += `<div><b>Bot:</b> ${data.answer}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
  }
});
