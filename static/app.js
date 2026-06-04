document.addEventListener("DOMContentLoaded", () => {
    const themeToggle = document.getElementById("theme-toggle");
    const htmlElement = document.documentElement;
    const chatForm = document.getElementById("chat-form");
    const messageInput = document.getElementById("message-input");
    const chatHistory = document.getElementById("chat-history");

    // Get User Name from prompt dialog on page load
    let userName = localStorage.getItem("user_name");
    if (!userName) {
        userName = prompt("Welcome to AI Therapist! Please enter your name:", "Guest");
        if (!userName || !userName.trim()) {
            userName = "Guest";
        }
        localStorage.setItem("user_name", userName);
    }

    // Update username elements in UI
    const userProfileNameEl = document.getElementById("user-profile-name");
    if (userProfileNameEl) {
        userProfileNameEl.textContent = userName;
    }
    const welcomeBannerText = document.getElementById("welcome-banner-text");
    if (welcomeBannerText) {
        welcomeBannerText.innerHTML = `Welcome, <strong>${escapeHtml(userName)}</strong>! A safe, confidential space to find emotional support and mental clarity. Take a slow, deep breath, and express your thoughts freely in Arabic or English.`;
    }

    // Maintain conversational chat session ID
    let chatSessionId = localStorage.getItem("chat_session_id");
    if (!chatSessionId) {
        chatSessionId = 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
        localStorage.setItem("chat_session_id", chatSessionId);
    }

    // Initialize Theme
    const savedTheme = localStorage.getItem("theme") || "dark";
    htmlElement.setAttribute("data-theme", savedTheme);
    updateThemeIcon(savedTheme);

    // Set welcome message timestamp dynamically
    const welcomeTimestamp = document.getElementById("welcome-timestamp");
    if (welcomeTimestamp) {
        welcomeTimestamp.textContent = formatTime(new Date());
    }

    // Historical Mood Scores for the SVG line chart
    let moodScores = [50, 58, 48, 62, 70];
    updateMoodChart();

    // Mindfulness Quotes rotation
    const MINDFULNESS_QUOTES = [
        "\"Quiet the mind and the soul will speak. Breathing in, I calm body and mind. Breathing out, I smile.\"",
        "\"Feelings come and go like clouds in a windy sky. Conscious breathing is my anchor.\"",
        "\"You don't have to control your thoughts. You just have to stop letting them control you.\"",
        "\"Do not dwell in the past, do not dream of the future, concentrate the mind on the present moment.\"",
        "\"The present moment is filled with joy and happiness. If you are attentive, you will see it.\"",
        "\"Breathe in experience, breathe out poetry. You are doing the best you can.\""
    ];

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
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    message: messageText,
                    session_id: chatSessionId
                })
            });

            removeElement(typingIndicator);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "An error occurred on the server.");
            }

            const data = await response.json();
            
            // 4. Add Bot Response to Chat History
            addBotMessage(data);

            // 5. Dynamically update right-hand Insights Panel
            updateInsightsPanel(data);

            // 6. Rotate Mindfulness quote
            rotateMindfulnessQuote();

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
        hours = hours ? hours : 12;
        minutes = minutes < 10 ? '0' + minutes : minutes;
        return `${hours}:${minutes} ${ampm}`;
    }

    // Helper: Scroll Chat to Bottom
    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // Helper: Remove element
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

        // Set marked options for breaks
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
                    <span class="badge badge-intent">
                        🧠 Intent: ${formatIntent(data.intent)}
                    </span>
                    <span class="badge badge-lang">
                        🌐 Language: ${capitalize(data.language)}
                    </span>
                </div>
        `;

        // If references are available, add references drawer
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

        // Wire up toggle logic for references drawer
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

    // Dynamically Update Insights Panel
    function updateInsightsPanel(data) {
        // Update Detected Intent
        const intentEl = document.getElementById("detected-intent-val");
        if (intentEl) {
            intentEl.textContent = formatIntent(data.intent);
        }

        // Calculate emotion percentages based on API response
        let sadness = 10;
        let anxiety = 15;
        let fatigue = 20;
        let joy = 12;

        const currentEmotion = data.emotion.toLowerCase();
        const confVal = Math.round(data.emotion_confidence * 100);

        if (currentEmotion === "sadness") {
            sadness = confVal;
            anxiety = Math.round(confVal * 0.7);
            fatigue = Math.round(confVal * 0.8);
            joy = Math.round(100 - confVal);
        } else if (currentEmotion === "fear" || currentEmotion === "anxiety" || currentEmotion === "anger") {
            anxiety = confVal;
            sadness = Math.round(confVal * 0.6);
            fatigue = Math.round(confVal * 0.7);
            joy = Math.round(100 - confVal);
        } else if (currentEmotion === "joy" || currentEmotion === "love" || currentEmotion === "surprise") {
            joy = confVal;
            sadness = Math.max(5, Math.round(100 - confVal * 1.2));
            anxiety = Math.max(8, Math.round(100 - confVal * 1.1));
            fatigue = Math.max(10, Math.round(100 - confVal * 0.9));
        }

        // Clip values between 2% and 100
        sadness = Math.max(2, Math.min(100, sadness));
        anxiety = Math.max(2, Math.min(100, anxiety));
        fatigue = Math.max(2, Math.min(100, fatigue));
        joy = Math.max(2, Math.min(100, joy));

        // Update progress bars
        setProgressBar("sadness", sadness);
        setProgressBar("anxiety", anxiety);
        setProgressBar("fatigue", fatigue);
        setProgressBar("joy", joy);

        // Update Mood Score Trend
        let delta = 0;
        if (currentEmotion === "joy" || currentEmotion === "love") delta = 8;
        else if (currentEmotion === "neutral") delta = 1;
        else if (currentEmotion === "surprise") delta = 3;
        else delta = -7;

        let currentMood = moodScores[moodScores.length - 1] + delta;
        currentMood = Math.max(15, Math.min(99, currentMood));
        
        moodScores.push(currentMood);
        if (moodScores.length > 6) {
            moodScores.shift();
        }

        const moodBadge = document.getElementById("mood-score-val");
        if (moodBadge) {
            moodBadge.textContent = currentMood;
            if (currentMood >= 70) {
                moodBadge.style.color = "#10b981";
                moodBadge.style.background = "rgba(16, 185, 129, 0.1)";
            } else if (currentMood >= 45) {
                moodBadge.style.color = "#f59e0b";
                moodBadge.style.background = "rgba(245, 158, 11, 0.1)";
            } else {
                moodBadge.style.color = "#ef4444";
                moodBadge.style.background = "rgba(239, 68, 68, 0.1)";
            }
        }

        updateMoodChart();
    }

    function setProgressBar(id, value) {
        const pctEl = document.getElementById(`${id}-pct`);
        const barEl = document.getElementById(`${id}-bar`);
        if (pctEl) pctEl.textContent = `${value}%`;
        if (barEl) barEl.style.width = `${value}%`;
    }

    // Dynamic SVG line charting helper
    function updateMoodChart() {
        const chartArea = document.getElementById("mood-chart-area");
        const chartLine = document.getElementById("mood-chart-line");
        if (!chartArea || !chartLine) return;

        const xStep = 20;
        const points = moodScores.map((score, idx) => {
            const x = idx * xStep;
            const y = 38 - (score / 100) * 33;
            return { x, y };
        });

        const linePath = points.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(" ");
        chartLine.setAttribute("d", linePath);

        const areaPath = `${linePath} L 100 40 L 0 40 Z`;
        chartArea.setAttribute("d", areaPath);
    }

    // Rotate mindfulness quote card
    function rotateMindfulnessQuote() {
        const quoteEl = document.querySelector(".mindfulness-quote");
        if (quoteEl) {
            const randomQuote = MINDFULNESS_QUOTES[Math.floor(Math.random() * MINDFULNESS_QUOTES.length)];
            quoteEl.style.opacity = 0;
            setTimeout(() => {
                quoteEl.textContent = randomQuote;
                quoteEl.style.opacity = 1;
            }, 250);
        }
    }

    // Helper: Format intent name
    function formatIntent(intent) {
        if (!intent) return "General";
        return intent
            .replace(/_/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
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
