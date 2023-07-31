function checkEnter(event) {
	if (event.key === "Enter") {
		inputReceived(event.target);
	}
}

function inputReceived(textboxElement) {
	const inputText = textboxElement.value;
	textboxElement.value = "";
	setRandomBoxColours();
	
	resetBoxHighlights();
	highlightBox();
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

function highlightBox() {
	let randBox = Math.floor(Math.random() * 3);
	let boxId = `box-${randBox}-outer`;
	setTimeout(() => {
		document.getElementById(boxId).classList.add('show-border');
	}, 50);
}


document.addEventListener("DOMContentLoaded", setRandomBoxColours);