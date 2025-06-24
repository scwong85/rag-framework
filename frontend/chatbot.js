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
    let sourcesHTML = '';
    if (data.sources && data.sources.length > 0) {
        const sourcesList = data.sources.map(src => 
            `<li><a href="${src}" target="_blank" class="text-blue-500 underline">${src}</a></li>`
        ).join('');
        sourcesHTML = `<div><strong>Read more at:</strong><ul class="ml-4 list-disc">${sourcesList}</ul></div>`;
    }
    chatBox.innerHTML += `
      ${sourcesHTML}
    `;
    chatBox.scrollTop = chatBox.scrollHeight;
  }
});
