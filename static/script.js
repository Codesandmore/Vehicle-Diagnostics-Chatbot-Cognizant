document.addEventListener("DOMContentLoaded", function () {
  const chatBox = document.getElementById("chat-box");
  const userInput = document.getElementById("user-input");
  const sendButton = document.getElementById("send-button");

  // Add welcome message
  addMessage(
    "Welcome to the Vehicle Diagnostics Chatbot! Ask me anything about your vehicle.",
    "bot"
  );

  // Send message function
  function sendMessage() {
    const message = userInput.value.trim();
    if (message === "") return;

    // Add user message to chat
    addMessage(message, "user");
    userInput.value = "";

    // Disable send button and show typing indicator
    sendButton.disabled = true;
    addMessage("Typing...", "bot", true);

    // Send message to backend
    fetch("/send_message", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: message }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Remove typing indicator
        removeTypingIndicator();

        // Add bot response
        addMessage(data.reply, "bot");

        // Re-enable send button
        sendButton.disabled = false;
      })
      .catch((error) => {
        console.error("Error:", error);
        removeTypingIndicator();
        addMessage("Sorry, I encountered an error. Please try again.", "bot");
        sendButton.disabled = false;
      });
  }

  // Add message to chat box
  function addMessage(text, sender, isTyping = false) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}-message`;
    if (isTyping) {
      messageDiv.className += " typing";
      messageDiv.id = "typing-indicator";
    }
    messageDiv.textContent = text;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  // Remove typing indicator
  function removeTypingIndicator() {
    const typingIndicator = document.getElementById("typing-indicator");
    if (typingIndicator) {
      typingIndicator.remove();
    }
  }

  // Event listeners
  sendButton.addEventListener("click", sendMessage);

  userInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      sendMessage();
    }
  });

  // Focus on input field
  userInput.focus();
});
