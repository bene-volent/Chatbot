document.addEventListener('DOMContentLoaded', function () {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const resetBtn = document.getElementById('reset-btn');
    const sendBtn = document.getElementById('send-btn');
    const buttonText = document.querySelector('.button-text');
    const buttonSpinner = document.querySelector('.button-spinner');
    const statusIndicator = document.getElementById('status-indicator');

    // WebSocket connection
    let ws;
    let clientId = 'client_' + Date.now();
    let isConnected = false;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;

    // Keep track of message elements for potential updates
    const messageElements = {};
    let lastBotMessageId = null;
    let currentBotMessageElement = null;
    let streamingInProgress = false;
    
    // Connect WebSocket
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${clientId}`;
        
        ws = new WebSocket(wsUrl);
        
        ws.onopen = function() {
            console.log('WebSocket connected');
            isConnected = true;
            reconnectAttempts = 0;
            updateStatus('ready');
        };
        
        ws.onclose = function() {
            console.log('WebSocket disconnected');
            isConnected = false;
            
            if (reconnectAttempts < maxReconnectAttempts) {
                reconnectAttempts++;
                console.log(`Attempting to reconnect (${reconnectAttempts}/${maxReconnectAttempts})...`);
                setTimeout(connectWebSocket, 2000); // Reconnect after 2 seconds
            } else {
                updateStatus('error');
                addMessage('Connection lost. Please refresh the page.', false);
            }
        };
        
        ws.onerror = function(err) {
            console.error('WebSocket error:', err);
            updateStatus('error');
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            switch(data.type) {
                case 'start':
                    // Create a new message element if this is not a continuation
                    if (!data.is_continuation) {
                        currentBotMessageElement = addMessage('', false);
                        lastBotMessageId = currentBotMessageElement.id;
                    } else {
                        // For continuation, use the last bot message
                        currentBotMessageElement = document.getElementById(lastBotMessageId);
                    }
                    hideLoading();
                    streamingInProgress = true;
                    
                    // Add streaming class to show cursor animation
                    if (messageElements[lastBotMessageId]) {
                        messageElements[lastBotMessageId].classList.add('streaming');
                    }
                    break;
                    
                case 'token':
                    if (currentBotMessageElement && messageElements[lastBotMessageId]) {
                        // Append the new token to the message with proper spacing
                        const content = messageElements[lastBotMessageId];
                        const currentText = content.textContent;
                        
                        // Fix spacing issues
                        if (currentText && !currentText.endsWith(' ') && 
                            !data.token.startsWith(' ') && 
                            ![',', '.', '!', '?', ':', ';', ')', ']', '}'].includes(data.token)) {
                            content.textContent = currentText + ' ' + data.token;
                        } else {
                            content.textContent = currentText + data.token;
                        }
                        
                        // Scroll to bottom to follow the new content
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    }
                    break;
                    
                case 'end':
                    streamingInProgress = false;
                    if (currentBotMessageElement && messageElements[lastBotMessageId]) {
                        // Remove streaming class to stop cursor animation
                        messageElements[lastBotMessageId].classList.remove('streaming');
                        
                        // Add a brief highlight effect to show completion
                        currentBotMessageElement.classList.add('updated');
                        setTimeout(() => {
                            currentBotMessageElement.classList.remove('updated');
                        }, 1000);
                    }
                    break;
                    
                case 'error':
                    hideLoading();
                    updateStatus('error');
                    addMessage(`Error: ${data.error}`, false);
                    streamingInProgress = false;
                    
                    // Reset status after a delay
                    setTimeout(() => {
                        updateStatus('ready');
                    }, 3000);
                    break;
            }
        };
    }
    
    // Connect on page load
    connectWebSocket();

    // Update status display
    function updateStatus(state) {
        statusIndicator.className = 'status ' + state;
        switch (state) {
            case 'ready':
                statusIndicator.textContent = 'Ready';
                break;
            case 'thinking':
                statusIndicator.textContent = 'Thinking...';
                break;
            case 'error':
                statusIndicator.textContent = 'Error';
                break;
        }
    }

    // Modified addMessage function to track message elements
    function addMessage(content, isUser = false) {
        const messageId = 'msg_' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user' : 'bot');
        messageDiv.id = messageId;

        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        messageContent.textContent = content;

        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Store reference to bot messages for potential updates
        if (!isUser) {
            lastBotMessageId = messageId;
            messageElements[messageId] = messageContent;
        }

        return messageDiv;
    }
    
    // Function to show loading animation
    function showLoading() {
        updateStatus('thinking');
        buttonText.classList.add('hidden');
        buttonSpinner.classList.remove('hidden');
        sendBtn.disabled = true;
        userInput.disabled = true;

        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('loading');
        loadingDiv.id = 'loading-indicator';

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.classList.add('loading-dot');
            loadingDiv.appendChild(dot);
        }

        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to hide loading animation
    function hideLoading() {
        updateStatus('ready');
        buttonText.classList.remove('hidden');
        buttonSpinner.classList.add('hidden');
        sendBtn.disabled = false;
        userInput.disabled = false;

        const loadingIndicator = document.getElementById('loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
    }

    // Handle form submission with continuation detection
    chatForm.addEventListener('submit', function (e) {
        e.preventDefault();

        const message = userInput.value.trim();
        if (!message) return;

        // Check if this is a continuation request
        const isContinuation = /^(continue|complete it|go on|proceed|keep going)$/i.test(message);

        // Check if WebSocket is connected
        if (!isConnected) {
            addMessage('Connection lost. Attempting to reconnect...', false);
            connectWebSocket();
            return;
        }

        if (streamingInProgress) {
            addMessage('Please wait for the current message to complete.', false);
            return;
        }

        // Add user message to chat
        addMessage(message, true);

        // Clear input and focus it
        userInput.value = '';
        userInput.focus();

        // Show loading animation
        showLoading();

        // Send message via WebSocket
        ws.send(JSON.stringify({
            message: message,
            session_id: sessionId,
            continue_last: isContinuation
        }));
    });

    // Handle reset button
    resetBtn.addEventListener('click', async function () {
        try {
            const formData = new FormData();
            formData.append('session_id', sessionId);

            const response = await fetch('/reset', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            // Clear chat messages except for the initial greeting
            while (chatMessages.children.length > 1) {
                chatMessages.removeChild(chatMessages.lastChild);
            }

            updateStatus('ready');

        } catch (error) {
            console.error('Error:', error);
            updateStatus('error');
            addMessage('Sorry, there was an error resetting the conversation.');

            // Reset status after a delay
            setTimeout(() => {
                updateStatus('ready');
            }, 3000);
        }
    });

    // Initialize with ready status
    updateStatus('ready');
});