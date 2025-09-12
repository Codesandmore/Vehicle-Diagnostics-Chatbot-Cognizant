document.addEventListener("DOMContentLoaded", function () {
  const chatBox = document.getElementById("chat-box");
  const userInput = document.getElementById("user-input");
  const sendButton = document.getElementById("send-button");
  const imageButton = document.getElementById("image-button");
  const imageInput = document.getElementById("image-input");
  const imagePreview = document.getElementById("image-preview");
  const previewImg = document.getElementById("preview-img");
  const removeImageButton = document.getElementById("remove-image");

  let selectedImage = null;

  // Add welcome message
  addMessage(
    "Welcome to the Vehicle Diagnostics Chatbot! Ask me anything about your vehicle or upload an image for analysis.",
    "bot"
  );

  // Send message function
  function sendMessage() {
    const message = userInput.value.trim();
    if (message === "" && !selectedImage) return;

    // Create FormData for sending both text and image
    const formData = new FormData();
    formData.append("message", message);

    if (selectedImage) {
      formData.append("image", selectedImage);
      // Add user message with image to chat
      addMessageWithImage(
        message || "Uploaded an image",
        "user",
        selectedImage
      );
    } else {
      // Add user message to chat
      addMessage(message, "user");
    }

    userInput.value = "";
    clearImagePreview();

    // Disable send button and show typing indicator
    sendButton.disabled = true;
    addMessage("Analyzing...", "bot", true);

    // Send message to backend
    fetch("/send_message", {
      method: "POST",
      body: formData,
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

    // Convert markdown-style formatting to HTML for bot messages
    if (sender === "bot" && !isTyping) {
      messageDiv.innerHTML = formatMarkdown(text);
    } else {
      messageDiv.textContent = text;
    }

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  // Convert markdown-style formatting to HTML
  function formatMarkdown(text) {
    let formatted = text;

    // First, escape any existing HTML to prevent injection
    formatted = formatted.replace(/</g, "&lt;").replace(/>/g, "&gt;");

    // Convert **bold** to <strong>
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Convert *italic* (but not if it's part of **bold**)
    formatted = formatted.replace(/(?<!\*)\*([^*]+?)\*(?!\*)/g, "<em>$1</em>");

    // Convert line breaks to <br>
    formatted = formatted.replace(/\n/g, "<br>");

    // Handle bullet points with nested structure
    const lines = formatted.split("<br>");
    let inList = false;
    let processedLines = [];

    for (let i = 0; i < lines.length; i++) {
      let line = lines[i].trim();

      // Check if line starts with bullet point
      if (line.match(/^\*\s+(.+)/)) {
        if (!inList) {
          processedLines.push("<ul>");
          inList = true;
        }
        // Extract content after the asterisk
        const content = line.replace(/^\*\s+/, "");
        processedLines.push(`<li>${content}</li>`);
      } else {
        // If we were in a list and this line doesn't start with *, close the list
        if (inList && line !== "") {
          processedLines.push("</ul>");
          inList = false;
        }
        if (line !== "") {
          processedLines.push(line);
        }
      }
    }

    // Close list if we end while still in one
    if (inList) {
      processedLines.push("</ul>");
    }

    formatted = processedLines.join("<br>");

    // Clean up extra <br> tags around lists
    formatted = formatted.replace(/<br><ul>/g, "<ul>");
    formatted = formatted.replace(/<\/ul><br>/g, "</ul>");
    formatted = formatted.replace(/<br><li>/g, "<li>");
    formatted = formatted.replace(/<\/li><br>/g, "</li>");

    // Remove empty <br> tags
    formatted = formatted.replace(/(<br>){2,}/g, "<br><br>");

    return formatted;
  }

  // Add message with image to chat box
  function addMessageWithImage(text, sender, imageFile) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}-message`;

    if (text) {
      const textDiv = document.createElement("div");
      textDiv.textContent = text;
      messageDiv.appendChild(textDiv);
    }

    const imgElement = document.createElement("img");
    imgElement.className = "message-image";
    imgElement.src = URL.createObjectURL(imageFile);
    imgElement.alt = "Uploaded image";
    messageDiv.appendChild(imgElement);

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  // Handle image selection
  function handleImageSelect(event) {
    const file = event.target.files[0];
    if (file && file.type.startsWith("image/")) {
      selectedImage = file;
      previewImg.src = URL.createObjectURL(file);
      imagePreview.style.display = "block";
    }
  }

  // Clear image preview
  function clearImagePreview() {
    selectedImage = null;
    imagePreview.style.display = "none";
    previewImg.src = "";
    imageInput.value = "";
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
  imageButton.addEventListener("click", () => imageInput.click());
  imageInput.addEventListener("change", handleImageSelect);
  removeImageButton.addEventListener("click", clearImagePreview);

  userInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      sendMessage();
    }
  });

  // Focus on input field
  userInput.focus();
});
