---
---
{% include_relative colour-model.js %}
var colourVocab = {{ site.data.vocab | jsonify }};
var listenerModelWeights = {{ site.data.listener_model | jsonify }};

function checkEnter(event) {
	if (event.key === "Enter") {
		inputReceived(event.target);
	}
}

function inputReceived(textboxElement) {
	const inputText = textboxElement.value;
	if (inputText !== "") {
		processInput(inputText);
		textboxElement.value = "";
	}
}

function processInput(inputText) {
	resetBoxHighlights();
	
	const inputTokens = Tokeniser.heuristicEndingTokeniser(inputText);
	const inputIndices = inputTokens.map(t => tokenToIndice(t, colourVocab));
	
	function tokenToIndice(token, vocab) {
		return (token in vocab) ? vocab[token] : vocab['<unk>']
	}
	const prediction = model.predict(inputIndices, currColourVectors);
	highlightBox(prediction);
}

function nextClicked() {
	setRandomBoxColours();
	resetBoxHighlights();
}

function generateRandomColour() {
	// Use HSL where L is fixed to 50
	let h = Math.floor(Math.random() * 360);
	let s = Math.floor(Math.random() * 100);
	let l = 50;
	
	return [h, s, l]
}

function setRandomBoxColours() {
	const randomColour0 = generateRandomColour();
	const randomColour1 = generateRandomColour();
	const randomColour2 = generateRandomColour();
	
	changeBoxColour(document.getElementById("box-0"), randomColour0);
	changeBoxColour(document.getElementById("box-1"), randomColour1);
	changeBoxColour(document.getElementById("box-2"), randomColour2);
	
	currColours[0] = randomColour0;
	currColours[1] = randomColour1;
	currColours[2] = randomColour2;
	currColourVectors = ColourModel.vectorizeColours(currColours);
}

function getHslCssString(h, s, l) {
	return `hsl(${h} ${s}% ${l}%`
}

function changeBoxColour(boxElement, colour) {
	const colourString = getHslCssString(colour[0], colour[1], colour[2]);
	boxElement.style.setProperty('background-color', colourString);
}

function resetBoxHighlights() {
	document.getElementById('box-0-outer').classList.remove('show-border');
	document.getElementById('box-1-outer').classList.remove('show-border');
	document.getElementById('box-2-outer').classList.remove('show-border');
	
	document.getElementById('box-0-outer').style.setProperty('border-color', 'transparent');
	document.getElementById('box-1-outer').style.setProperty('border-color', 'transparent');
	document.getElementById('box-2-outer').style.setProperty('border-color', 'transparent');
}

function highlightBox(prediction) {
	let boxId = `box-${prediction}-outer`;
	setTimeout(() => {
		document.getElementById(boxId).classList.add('show-border');
	}, 50);
}

document.addEventListener("DOMContentLoaded", setRandomBoxColours);
var currColours = [];
var currColourVectors;
const model = new ColourModel({vocabDim: 962, hiddenDim: 100, weights: listenerModelWeights});