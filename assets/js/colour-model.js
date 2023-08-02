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

// Adapted from https://github.com/futurulus/coop-nets
function vectorize_all(colors) {
	// `colors` is an array of dimension (3, 3), where the first dimension is a colour sample and the second is the HSL value of the colour
    colors = [colors];
    if (colors[0].length !== 3 || colors[0][0].length !== 3) {
        throw new Error("Invalid colors shape.");
    }

    const ranges = [361.0, 101.0, 101.0];
    const color_0_1 = colors.map(color =>
        color.map(c => c.map((v, i) => v / (ranges[i] - 1.0)))
    );

    // Using a Fourier representation causes colors at the boundary of the
    // space to behave as if the space is toroidal: red = 255 would be
    // about the same as red = 0. We don't want this...
    const xyz = color_0_1[0].map(c => [c[0], c[1] / 2.0, c[2] / 2.0]);
	
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