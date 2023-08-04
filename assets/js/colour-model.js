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
		
		return scores;
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
		colours = tf.tensor2d(colours);
	
		if (colours.shape[0] !== 3 || colours.shape[0] !== 3) {
			throw new Error("Invalid colours shape.");
		}

		const ranges = tf.tensor1d([361.0, 101.0, 101.0]);
		const colour_0_1 = colours.div(ranges.sub(tf.scalar(1.0)));

		// Using a Fourier representation causes colours at the boundary of the
		// space to behave as if the space is toroidal: red = 255 would be
		// about the same as red = 0. We don't want this...
		const xyz = colour_0_1.div(tf.tensor1d([1, 2, 2]));
    
		// Meshgrid operations on three [0, 1, 2] vectors
		const gz = tf.broadcastTo(tf.range(0, 3), [3, 3, 3]);
		const gx = gz.transpose([0, 2, 1]);
		const gy = gz.transpose([2, 1, 0]);
		
			// Outer multiplication between xyz and gx, gy, gz respectively
		// Get first column (3,) from xyz (3,3), and outerProduct with gx
		const xyz_x = xyz.slice([0, 0], [3, 1]).squeeze();
		const argx = tf.einsum('i,jkl->ijkl', xyz_x, gx);
		
		// Get second column (3,) from xyz (3,3), and outerProduct with gy
		const xyz_y = xyz.slice([0, 1], [3, 1]).squeeze();
		const argy = tf.einsum('i,jkl->ijkl', xyz_y, gy);
		
		// Get third column (3,) from xyz (3,3), and outerProduct with gz
		const xyz_z = xyz.slice([0, 2], [3, 1]).squeeze();
		const argz = tf.einsum('i,jkl->ijkl', xyz_z, gz);
			
		const argTf = argx.add(argy).add(argz);
		
		// Use mathjs library here because TensorflowJS exp() does not support complex numbers
		const arg = math.matrix(argTf.arraySync());
		const reprComplex = math.map(math.multiply(math.multiply(math.complex(0, -2), math.pi), math.mod(arg, 1)), math.exp);
		
		const reprComplexTf = tf.complex(math.re(reprComplex)._data, math.im(reprComplex)._data);
		const reshappedRepr = reprComplexTf.transpose([0, 2, 1, 3]);
		const result = tf.concat([tf.real(reshappedRepr).reshape([3, 27]), tf.imag(reshappedRepr).reshape([3, 27])], 1);
		const normalized = result.mul(100).round().div(100); // Round to 2 digits
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
		const colourVectors = currColourVectors.expandDims(); // Add a new dimension at axis 0 to make this a batch
		const predictedScores = this.model.predict([tf.tensor([inputIndices]), colourVectors]);
		predictedScores.print(); // Do we flatten the predictedScores?
		return predictedScores.argMax(1).dataSync();
	}
}