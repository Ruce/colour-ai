---
---
{% include_relative colour-model.js %}
var colourVocab = {{ site.data.vocab | jsonify }};
var listenerModelWeights = {{ site.data.listener_model| jsonify }};

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
	const inputTokens = Tokeniser.heuristicEndingTokeniser(inputText);
	const inputIndices = inputTokens.map(t => tokenToIndice(t, colourVocab));
	
	function tokenToIndice(token, vocab) {
		return (token in vocab) ? vocab[token] : vocab['<unk>']
	}
	
	console.log(inputIndices);
	
	let prediction = predict(inputIndices);
	console.log(prediction);
	//let prediction = Math.floor(Math.random() * 3);
		
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
	
	changeBoxColour(document.getElementById("box-0"), randomColour0);
	changeBoxColour(document.getElementById("box-1"), randomColour1);
	changeBoxColour(document.getElementById("box-2"), randomColour2);
	
	currColours[0] = randomColour0;
	currColours[1] = randomColour1;
	currColours[2] = randomColour2;
	currColourVectors = vectorizeColours(currColours)._data;
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

function getTensor() {
	const embeddingsWeights = tf.tensor(listenerModelWeights.word_embeddings.weight);
	
	const embedding = tf.layers.embedding({inputDim: 962, outputDim: 100});
	console.log(embedding.apply(tf.tensor([[0], [1]])));
}

class ScoreLayer extends tf.layers.Layer {
	constructor(args) {
		super({});
		this.colourVectorDim = args.colourVectorDim;
	}
	
	computeOutputShape(inputShape) {
		return [];
	}
	
	call(inputs, kwargs) {
		const cVec = inputs[0];
		const rep = inputs[1];
		const covVector = inputs[2];
		const covMatrix = tf.reshape(covVector, [covVector.shape[0], this.colourVectorDim, this.colourVectorDim]); // Resulting shape (n, 54, 54)
		
		// Swap cVec's 0th and 1st dimensions so that we can broadcast subtract against rep, i.e. (f - mu) for each colour
		const cv = cVec.transpose([1, 0, 2]); // Resulting shape (3, n, 54)
		const delta = cv.sub(rep).transpose([1, 2, 0]); // Resulting shape (n, 54, 3)
		const deltaT = delta.transpose([0, 2, 1]); // Resulting shape (n, 3, 54)
		const scoresMatrix = deltaT.matMul(covMatrix).matMul(delta); // Resulting shape (n, 3, 3)
		 
		 // We are only interested in the diagonal of scoresMatrix
		const diagIndices = tf.range(0, 9, 4, 'int32');
		const scoresLogits = scoresMatrix.reshape([scoresMatrix.shape[0], 9]).gather(diagIndices, 1).mul(tf.scalar(-1)); // Resulting shape (n, 3)
		const scores = scoresLogits.softmax();
				
		/**
		const repMatrix = tf.reshape(rep, [rep.shape[0], rep.shape[1], 1]); // Unsqueeze last dimension of rep for later matrix operations, resulting shape (n, 54, 1)
		const scores = []
		for (let i = 0; i < 3; i++) {
			let cv = cVec.slice([0, i, 0], [cVec.shape[0], i+1, this.colourVectorDim]).reshape([cVec.shape[0], this.colourVectorDim, 1]); // Resulting shape (n, 54, 1)
			let delta = cv.sub(repMatrix); // Resulting shape (n, 54, 1)
			let deltaT = delta.transpose([0, 2, 1]); // Resulting shape (n, 1, 54)
			let score = tf.matMul(tf.matMul(deltaT, covMatrix), delta); // Resulting shape (n, 1, 1)
			scores.push(score.squeeze());
		}
		return tf.stack(scores, 1);**/
		return scores
	}
	
	getClassName() { return 'ColourScores'; }
}

function initialiseModel(vocabDim, hiddenDim) {
	const colourVectorDim = 54;
	const embeddingsWeights = tf.tensor(listenerModelWeights.word_embeddings.weight);
	const lstmInputWeights = tf.transpose(tf.tensor(listenerModelWeights.lstm.weight_ih_l0));
	const lstmHiddenWeights = tf.transpose(tf.tensor(listenerModelWeights.lstm.weight_hh_l0));
	const lstmInputBias = tf.tensor(listenerModelWeights.lstm.bias_ih_l0);
	const lstmHiddenBias = tf.tensor(listenerModelWeights.lstm.bias_hh_l0);
	const lstmBias = lstmInputBias.add(lstmHiddenBias); // Pytorch defines the input and hidden biases separately, but Tensorflow only has one bias vector for LSTMs
	const descriptionRepWeights = tf.transpose(tf.tensor(listenerModelWeights.description_representation.weight));
	const descriptionRepBias = tf.tensor(listenerModelWeights.description_representation.bias);
	const covarianceWeights = tf.transpose(tf.tensor(listenerModelWeights.covariance.weight));
	const covarianceBias = tf.tensor(listenerModelWeights.covariance.bias);
	
	// Set all weight and bias initializers to 'zeros' for performance, as initializers run before specified weights are set
	const embedding = tf.layers.embedding({inputDim: vocabDim, outputDim: hiddenDim, embeddingsInitializer: 'zeros', weights: [embeddingsWeights]});
	const lstm = tf.layers.lstm({units: hiddenDim, returnSequences: false, returnState: true, kernelInitializer: 'zeros', recurrentInitializer: 'zeros', biasInitializer: 'zeros', weights: [lstmInputWeights, lstmHiddenWeights, lstmBias]});
	const descriptionRepresentation = tf.layers.dense({units: colourVectorDim, kernelInitializer: 'zeros', biasInitializer: 'zeros', weights: [descriptionRepWeights, descriptionRepBias]});
	const covariance = tf.layers.dense({units: colourVectorDim*colourVectorDim, kernelInitializer: 'zeros', biasInitializer: 'zeros', weights: [covarianceWeights, covarianceBias]});
	const scoreLayer = new ScoreLayer({colourVectorDim: colourVectorDim});
	
	const inputTokens = tf.input({shape: [null]});
	const cVec = tf.input({shape: [3, colourVectorDim]});
	const embedded = embedding.apply(inputTokens);
	const lstm_out = lstm.apply(embedded); // Returns three values: sequence_outputs, last_state_h, last_state_c
	const rep = descriptionRepresentation.apply(lstm_out[1]); // Resulting shape (n, 54)
	const covVector = covariance.apply(lstm_out[1]); // Resulting shape (n, 2916)
	const scores = scoreLayer.apply([cVec, rep, covVector]);
	
	const model = tf.model({inputs: [inputTokens, cVec], outputs: scores});
	
	return model;
}

function predict(inputIndices) {
	const predictedScores = model.predict([tf.tensor([inputIndices]), tf.tensor([currColourVectors])]);
	predictedScores.print(); // Do we flatten the predictedScores?
	return predictedScores.argMax(1).dataSync();
}

document.addEventListener("DOMContentLoaded", setRandomBoxColours);
var currColours = [];
var currColourVectors = [];
const model = initialiseModel(962, 100);