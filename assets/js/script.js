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
	let prediction = Math.floor(Math.random() * 3);
		
	resetBoxHighlights();
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
	
	currColours.colour0 = randomColour0;
	currColours.colour1 = randomColour1;
	currColours.colour2 = randomColour2;
	
	changeBoxColour(document.getElementById("box-0"), randomColour0);
	changeBoxColour(document.getElementById("box-1"), randomColour1);
	changeBoxColour(document.getElementById("box-2"), randomColour2);
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




function initialiseModel() {
	const model = tf.sequential();
	model.add(tf.layers.dense({units: 10, activation: 'relu', inputShape: [10]}));
	model.add(tf.layers.dense({units: 1, activation: 'linear'}));
}


async function runTensorflow() {
	const model = initialiseModel();
	
	// Prepare the model for training: Specify the loss and the optimizer.
	model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

	// TO DO: Modify test data to the new model architecture
	// Generate some synthetic data for training. (y = 2x - 1)
	const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
	const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

	// Train the model using the data.
	await model.fit(xs, ys, {epochs: 50});
	
	console.log(model.predict(tf.tensor2d([20], [1, 1])).dataSync());
}













document.addEventListener("DOMContentLoaded", setRandomBoxColours);
let currColours = {}
