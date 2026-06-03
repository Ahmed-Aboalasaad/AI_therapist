document.addEventListener("DOMContentLoaded", () => {
    const themeToggle = document.getElementById("theme-toggle");
    const htmlElement = document.documentElement;
    const chatForm = document.getElementById("chat-form");
    const messageInput = document.getElementById("message-input");
    const chatHistory = document.getElementById("chat-history");
    const sendBtn = document.getElementById("send-btn");

    // Initialize Theme
    const savedTheme = localStorage.getItem("theme") || "dark";
    htmlElement.setAttribute("data-theme", savedTheme);
    updateThemeIcon(savedTheme);

    // Set welcome message timestamp dynamically
    const welcomeTimestamp = document.getElementById("welcome-timestamp");
    if (welcomeTimestamp) {
        welcomeTimestamp.textContent = formatTime(new Date());
    }

    // Theme Toggle Handler
    themeToggle.addEventListener("click", () => {
        const currentTheme = htmlElement.getAttribute("data-theme");
        const newTheme = currentTheme === "dark" ? "light" : "dark";
        htmlElement.setAttribute("data-theme", newTheme);
        localStorage.setItem("theme", newTheme);
        updateThemeIcon(newTheme);
    });

    function updateThemeIcon(theme) {
        const icon = themeToggle.querySelector("i");
        if (theme === "dark") {
            icon.className = "fa-solid fa-sun";
        } else {
            icon.className = "fa-solid fa-moon";
        }
    }

    // Chat Submission Handler
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const messageText = messageInput.value.trim();
        if (!messageText) return;

        // Clear input and refocus
        messageInput.value = "";
        messageInput.focus();

        // 1. Add User Message to UI
        const timeString = formatTime(new Date());
        addUserMessage(messageText, timeString);

        // 2. Add Bot Typing Indicator
        const typingIndicator = addTypingIndicator();
        scrollToBottom();

        // 3. Call API
        try {
            // API connection endpoint URL /api/chat
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ message: messageText })
            });

            removeElement(typingIndicator);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "An error occurred on the server.");
            }

            const data = await response.json();
            
            // 4. Add Bot Response
            addBotMessage(data);

        } catch (error) {
            removeElement(typingIndicator);
            addErrorMessage(`Connection Error: ${error.message}`);
        }

        scrollToBottom();
    });

    // Helper: Format Time e.g. 10:30 AM
    function formatTime(date) {
        let hours = date.getHours();
        let minutes = date.getMinutes();
        const ampm = hours >= 12 ? 'PM' : 'AM';
        hours = hours % 12;
        hours = hours ? hours : 12; // the hour '0' should be '12'
        minutes = minutes < 10 ? '0' + minutes : minutes;
        return `${hours}:${minutes} ${ampm}`;
    }

    // Helper: Scroll Chat to Bottom
    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // Helper: Remove element with fade-out
    function removeElement(element) {
        if (element && element.parentNode) {
            element.remove();
        }
    }

    // UI: Add User Message
    function addUserMessage(text, time) {
        const row = document.createElement("div");
        row.className = "message-row user-row";

        row.innerHTML = `
            <div class="avatar user-avatar">
                <i class="fa-solid fa-user"></i>
            </div>
            <div class="message-card-wrapper">
                <div class="message-card">
                    <p>${escapeHtml(text)}</p>
                </div>
                <div class="message-meta">
                    <span class="timestamp">${time}</span>
                    <span class="read-status"><i class="fa-solid fa-check-double"></i></span>
                </div>
            </div>
        `;
        chatHistory.appendChild(row);
    }

    // UI: Add Typing Indicator
    function addTypingIndicator() {
        const row = document.createElement("div");
        row.className = "message-row bot-row typing-indicator-row";

        row.innerHTML = `
            <div class="avatar bot-avatar">
                <i class="fa-solid fa-robot"></i>
            </div>
            <div class="message-card-wrapper">
                <div class="message-card typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        chatHistory.appendChild(row);
        return row;
    }

    // UI: Add Bot Message
    function addBotMessage(data) {
        const row = document.createElement("div");
        row.className = "message-row bot-row";
        
        const timeString = formatTime(new Date());
        
        // Emotion emoji map
        const emotionEmoji = {
            "joy": "😊",
            "sadness": "😢",
            "anger": "😡",
            "fear": "😰",
            "surprise": "😲",
            "love": "🥰",
            "neutral": "😐",
            "anxiety": "😟"
        };
        const emoji = emotionEmoji[data.emotion.toLowerCase()] || "🤖";
        const confidencePct = Math.round(data.emotion_confidence * 100);

        // Set marked options for GFM and auto-linebreaks
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                breaks: true,
                gfm: true
            });
        }

        const formattedResponse = typeof marked !== 'undefined' 
            ? marked.parse(data.response) 
            : escapeHtml(data.response).replace(/\n/g, '<br>');

        // Construct HTML content
        let contentHtml = `
            <div class="message-card">
                <div class="response-text" dir="auto">
                    ${formattedResponse}
                </div>
                <div class="badges-container">
                    <span class="badge badge-emotion">
                        ${emoji} Emotion: ${capitalize(data.emotion)} (${confidencePct}%)
                    </span>
                    <span class="badge badge-lang">
                        🌐 Language: ${capitalize(data.language)}
                    </span>
                </div>
        `;

        // If references are available, add a togglable references drawer
        if (data.references && data.references.length > 0) {
            contentHtml += `
                <div class="references-wrapper">
                    <div class="references-header">
                        <i class="fa-solid fa-circle-info"></i> References (RAG Context)
                    </div>
                    <div class="references-content">
                        ${data.references.map((ref, idx) => `
                            <div class="ref-item" style="margin-bottom: 8px;">
                                <strong>[Source ${idx + 1}]</strong><br>
                                <em>"${escapeHtml(ref.snippet)}"</em>
                            </div>
                        `).join("")}
                    </div>
                </div>
            `;
        }

        contentHtml += `
            </div>
            <div class="message-meta">
                <span class="timestamp">${timeString}</span>
            </div>
        `;

        row.innerHTML = `
            <div class="avatar bot-avatar">
                <i class="fa-solid fa-robot"></i>
            </div>
            <div class="message-card-wrapper">
                ${contentHtml}
            </div>
        `;

        chatHistory.appendChild(row);

        // Wire up toggle logic for references drawer if it exists
        const refHeader = row.querySelector(".references-header");
        if (refHeader) {
            const refContent = row.querySelector(".references-content");
            refHeader.addEventListener("click", () => {
                refContent.classList.toggle("show");
                scrollToBottom();
            });
        }
    }

    // UI: Add Error Message
    function addErrorMessage(text) {
        const row = document.createElement("div");
        row.className = "message-row bot-row";
        const timeString = formatTime(new Date());

        row.innerHTML = `
            <div class="avatar bot-avatar" style="color: #ef4444; background: rgba(239, 68, 68, 0.1);">
                <i class="fa-solid fa-triangle-exclamation"></i>
            </div>
            <div class="message-card-wrapper">
                <div class="message-card" style="border-color: rgba(239, 68, 68, 0.2); background: rgba(239, 68, 68, 0.02); color: #ef4444;">
                    <p>${escapeHtml(text)}</p>
                </div>
                <div class="message-meta">
                    <span class="timestamp">${timeString}</span>
                </div>
            </div>
        `;
        chatHistory.appendChild(row);
    }

    // Helper: Capitalize String
    function capitalize(str) {
        if (!str) return "";
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    // Helper: Escape HTML to prevent XSS
    function escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, function(m) { return map[m]; });
    }
});
