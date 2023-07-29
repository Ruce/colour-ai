function checkEnter(event) {
	if (event.key === "Enter") {
		inputReceived(event.target);
	}
}

function inputReceived(textboxElement) {
	const inputText = textboxElement.value;
	textboxElement.value = "";
	setRandomBoxColours();
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
	console.log(randomColour0, randomColour1, randomColour2);
}

function getHslCssString(h, s, l) {
	return `hsl(${h} ${s}% ${l}%`
}

function changeBoxColour(boxElement, colour) {
	const colourString = getHslCssString(colour[0], colour[1], colour[2]);
	boxElement.style.setProperty('background-color', colourString);
}


document.addEventListener("DOMContentLoaded", setRandomBoxColours);