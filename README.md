# Colour AI

![Screenshot of main page](/assets/img/ColoursScreenshot.png?raw=true "Screenshot")

Describe one of the three displayed colours and see if the AI guesses the colour you picked!

**Try it at (https://ruce.github.io/colour-ai/)**

## About

A simple project that uses the models from my Master's thesis, _Representing Vagueness with Probabilistic Semantics_ (publication pending), which explores the linguistics of colours in a contextual setting. Based on the dataset and prior work by [Monroe et al. (2017)](https://aclanthology.org/Q17-1023/).

![Listener model architecture](/assets/img/ListenerModelArchitecture.png?raw=true "Listener Model")

The listener model comprises of an encoder and a scoring function to predict a colour from the input text. The encoder embeds the tokens, pushes them through an LSTM layer, and passes the final state over linear layers to generate a representation with mean $\mu$ and covariance $\Sigma$ in a high-dimensional colour space.

The scoring function compares the encoded representation against the Fourier-transformed vectors $f$ for the displayed colours, and calculates the log-probability of each colour being the target:

$$score = -(f-\mu)^{T}\Sigma(f-\mu)$$

Finally, an exponential softmax is applied over the scores of the three colours to predict the target colour.