let gameStarted = false;
let gameID = null
async function startGame() {
    const response = await fetch('/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    });
    const data = await response.json();
    gameID = data.ID;
    // Update the chat box with the initial state and additional info
    const chatBox = document.getElementById('humanInfoBox');
    chatBox.innerHTML += "<p>" + data.additionalInfo + "</p>";
    chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the bottom

    // Update the additional fields
    document.getElementById('startGameField').innerText = data.startGameField;
    document.getElementById('chatBox').innerText = ''; // Clear or set as needed

    gameStarted = true;
}

async function updateState() {
    if (!gameStarted) {
        document.getElementById('humanInfoBox').innerText = 'Start the game first!';
        return;
    }

    const userInput1 = document.getElementById('userInput1').value;
    const userInput2 = document.getElementById('userInput2').value;

    // Send the inputs and receive a response
    const response = await fetch('/update?gameID=' + gameID, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({userInput1, userInput2}),
    });

    const data = await response.json();
    const humanInfoBox = document.getElementById('humanInfoBox');
    const chatBox = document.getElementById('chatBox');
    humanInfoBox.scrollTop = humanInfoBox.scrollHeight; // Auto-scroll to the bottom
    humanInfoBox.innerHTML += "<p>"+data.humanInfo+"</p><hr/>"
    chatBox.innerHTML += "<p>"+data.llmInfo+"</p><hr/>"

    // Clear the input fields
    document.getElementById('userInput1').value = '';
    document.getElementById('userInput2').value = '';
}

async function submitMessage(event) {
    event.preventDefault();
    await updateState();
}

function redirectToGame() {
    document.querySelector('.start-container').style.display = 'none';
    document.querySelector('.container').style.display = 'flex';
    startGame(); // Automatically start the game when redirecting

    // Add the new game button to the top-right corner
    const newGameButton = document.createElement('button');
    newGameButton.className = 'new-game-button';
    newGameButton.innerText = 'Start New Game';
    newGameButton.onclick = () => location.reload(); // Reload to reset the game
    document.body.appendChild(newGameButton);
}