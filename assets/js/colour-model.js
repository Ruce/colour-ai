var Tokeniser = {};

Tokeniser.WORD_RE_STR = /(?:[a-z][a-z'\-_]+[a-z])|(?:[+\-]?\d+[,/.:-]\d+[+\-]?)|(?:[\w_]+)|(?:\.(?:\s*\.){1,})|(?:\*{1,})|(?:\S)/gi;
Tokeniser.ENDINGS = ['er', 'est', 'ish'];

Tokeniser.basicUnigramTokeniser = function(s, lower=true) {
	let words = s.match(Tokeniser.WORD_RE_STR);
	if (lower) {
		words = words.map(x => x.toLowerCase());
	}
	return words;
}

Tokeniser.heuristicEndingTokeniser = function(s, lower=true) {
	let words = Tokeniser.basicUnigramTokeniser(s, lower);
	return words.flatMap((w) => Tokeniser.heuristicSegmenter(w));
}

Tokeniser.heuristicSegmenter = function(word) {
	for (const ending of Tokeniser.ENDINGS) {
		if (word.endsWith(ending)) {
			return [word.slice(0, -ending.length), '+' + ending];
		}
	}
	return [word];
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

class ColourModel {
	constructor(args) {
		this.vocabDim = args.vocabDim;
		this.hiddenDim = args.hiddenDim;
		this.weights = args.weights;
		this.model = this.initialiseModel();
	}
	
	static colourVectorDim = 54;
	
	// Adapted from https://github.com/futurulus/coop-nets
	static vectorizeColours(colours) {
		// `colours` is an array of dimension (3, 3), where the first dimension is a colour sample and the second is the HSL value of the colour
		colours = [colours];
		if (colours[0].length !== 3 || colours[0][0].length !== 3) {
			throw new Error("Invalid colours shape.");
		}

		const ranges = [361.0, 101.0, 101.0];
		const colour_0_1 = colours.map(colour =>
			colour.map(c => c.map((v, i) => v / (ranges[i] - 1.0)))
		);

		// Using a Fourier representation causes colours at the boundary of the
		// space to behave as if the space is toroidal: red = 255 would be
		// about the same as red = 0. We don't want this...
		const xyz = colour_0_1[0].map(c => [c[0], c[1] / 2.0, c[2] / 2.0]);
		
		// Meshgrid operations on ax, ay, az
		const gx = Array(3).fill([Array(3).fill(0), Array(3).fill(1), Array(3).fill(2)]);
		const gy = [Array(3).fill(Array(3).fill(0)), Array(3).fill(Array(3).fill(1)), Array(3).fill(Array(3).fill(2))];
		const gz = Array(3).fill(Array(3).fill([0, 1, 2]));
		
		// Outer multiplication between xyz and gx, gy, gz respectively
		const argx = xyz.map(a => a[0]).map(x =>
			gx.map(dim0 => 
				dim0.map(dim1 =>
					dim1.map(dim2 => x*dim2)
				)
			)
		);
		const argy = xyz.map(a => a[1]).map(y =>
			gy.map(dim0 => 
				dim0.map(dim1 =>
					dim1.map(dim2 => y*dim2)
				)
			)
		);
		const argz = xyz.map(a => a[2]).map(z =>
			gz.map(dim0 => 
				dim0.map(dim1 =>
					dim1.map(dim2 => z*dim2)
				)
			)
		);
		
		const arg = math.matrix(math.add(math.add(argx, argy), argz));
		const reprComplex = swapAxes(math.map(math.multiply(math.multiply(math.complex(0, -2), math.pi), math.mod(arg, 1)), math.exp));
		
		function swapAxes(x) {
		  let swappedMatrix = [];
		  for (let i = 0; i < 3; i++) {
			let outerSlice = [];
			for (let j = 0; j < 3; j++) {
			  let innerSlice = math.subset(x, math.index(i, math.range(0, 3), j, math.range(0, 3)));
			  outerSlice.push(math.reshape(innerSlice, [3, 3]));
			}
			swappedMatrix.push(math.matrix(outerSlice));
		  }
		  swappedMatrix = math.matrix(swappedMatrix);
		  return swappedMatrix;
		}
	  
		const reshappedRepr = math.reshape(reprComplex, [3, 27]);
		const result = math.concat(math.re(reshappedRepr), math.im(reshappedRepr), 1);

		const normalized = math.map(result, v => roundValues(v, 2));
		function roundValues(v, decimals) {
			const roundedValue = math.round(v, decimals);
			return roundedValue === 0 ? 0 : roundedValue;
		}

		return normalized;
	}

	initialiseModel() {
		const colourVectorDim = ColourModel.colourVectorDim;
		
		const embeddingsWeights = tf.tensor(this.weights.word_embeddings.weight);
		const lstmInputWeights = tf.transpose(tf.tensor(this.weights.lstm.weight_ih_l0));
		const lstmHiddenWeights = tf.transpose(tf.tensor(this.weights.lstm.weight_hh_l0));
		const lstmInputBias = tf.tensor(this.weights.lstm.bias_ih_l0);
		const lstmHiddenBias = tf.tensor(this.weights.lstm.bias_hh_l0);
		const lstmBias = lstmInputBias.add(lstmHiddenBias); // Pytorch defines the input and hidden biases separately, but Tensorflow only has one bias vector for LSTMs
		const descriptionRepWeights = tf.transpose(tf.tensor(this.weights.description_representation.weight));
		const descriptionRepBias = tf.tensor(this.weights.description_representation.bias);
		const covarianceWeights = tf.transpose(tf.tensor(this.weights.covariance.weight));
		const covarianceBias = tf.tensor(this.weights.covariance.bias);
		
		// Set all weight and bias initializers to 'zeros' for performance, as initializers run before specified weights are set
		const embedding = tf.layers.embedding({inputDim: this.vocabDim, outputDim: this.hiddenDim, embeddingsInitializer: 'zeros', weights: [embeddingsWeights]});
		const lstm = tf.layers.lstm({units: this.hiddenDim, returnSequences: false, returnState: true, kernelInitializer: 'zeros', recurrentInitializer: 'zeros', biasInitializer: 'zeros', weights: [lstmInputWeights, lstmHiddenWeights, lstmBias]});
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

	predict(inputIndices, currColourVectors) {
		const predictedScores = this.model.predict([tf.tensor([inputIndices]), tf.tensor([currColourVectors])]);
		predictedScores.print(); // Do we flatten the predictedScores?
		return predictedScores.argMax(1).dataSync();
	}
}