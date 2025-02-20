document.addEventListener('DOMContentLoaded', function () {
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');

    sendButton.addEventListener('click', function () {
        const userMessage = userInput.value.trim();
        if (userMessage === '') return;

        chatbox.innerHTML += `<div class="message user">${userMessage}</div>`;
        userInput.value = '';

        fetch('/get_response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ user_input: userMessage }),
        })
            .then(response => response.json())
            .then(data => {
                chatbox.innerHTML += `<div class="message bot">${data.response}</div>`;
            })
            .catch(error => console.error('Error:', error));
    });
});
