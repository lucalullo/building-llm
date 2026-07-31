# Building LLM

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

A progressive educational project that shows how language models are built step by step, from a simple character-level statistical model to a small decoder-only Transformer.

The project begins with standard Python and introduces one new concept in each version. The goal is not to compete with industrial models, but to make tokenization, probability, training, neural networks, attention and autoregressive generation easier to understand.

## Project roadmap

The complete project evolves through five main stages:

1. statistical foundations;
2. neural foundations;
3. sequence models;
4. attention and Transformer architecture;
5. a small decoder-only language model.

Each stage introduces the concepts required by the next one while keeping the implementation understandable and verifiable.

![Building LLM project roadmap infographic](infographic.png)

## Current version

Version 2 implements a trainable character-level neural language model using NumPy.

It continues directly from Version 1:

- the same English training corpus is used;
- adjacent characters remain the input-target examples;
- characters are converted into numerical identifiers;
- input identifiers are represented as one-hot vectors;
- a trainable weight matrix replaces direct transition counts;
- softmax converts logits into probability distributions;
- cross-entropy measures the prediction error;
- gradient descent updates the model parameters;
- text is generated one character at a time using a local random seed.

The model contains 729 trainable parameters and learns to reduce the loss from approximately `3.2973` to `1.9461`.

The first versions are language models, but they are not yet large language models. The LLM label becomes appropriate later in the project, after the introduction of subword tokenization, a decoder-only Transformer, a meaningful parameter count and training on a substantial corpus.

## Current architecture

```text
English Training Text
          ↓
Adjacent Character Pairs
          ↓
Numerical Character IDs
          ↓
One-Hot Input Vectors
          ↓
Trainable Weight Matrix
          ↓
Logits and Softmax Probabilities
          ↓
Cross-Entropy Loss
          ↓
Gradient Descent
          ↓
Next-Character Sampling
          ↓
Generated Text
```

The model still uses one character as context. The difference is that next-character probabilities are now learned through optimization instead of being calculated directly from transition counts.

## Version 1 - Character Statistical Model

For every character in the corpus, the model counts which characters appeared immediately after it.

For example, the word `model` produces the following adjacent-character pairs:

```text
m → o
o → d
d → e
e → l
```

The transition counts are normalized into probabilities:

```text
P(next | current) = count(current → next) / total counts(current)
```

Generation is autoregressive:

1. read the current character;
2. sample the next character;
3. append it to the output;
4. use the sampled character as the new context;
5. repeat.

The random generator uses a local seed, making the example reproducible without modifying Python's global random state.

Open the folder:

[`v01-character-model`](v01-character-model/)

## Version 2 - Neural Character Model with NumPy

Version 2 preserves the character-level bigram structure introduced in Version 1, but replaces direct statistical counts with trainable parameters.

Each character is converted into an integer identifier and then represented as a one-hot vector.

For example:

```text
model → [15, 17, 7, 8, 14]
```

The complete input matrix has shape:

```text
(439, 27)
```

The model uses a trainable weight matrix with shape:

```text
(27, 27)
```

This produces:

```text
27 × 27 = 729 trainable parameters
```

Training follows this sequence:

1. calculate the logits;
2. convert the logits into probabilities with softmax;
3. calculate the cross-entropy loss;
4. calculate the gradients;
5. update the weights with gradient descent;
6. repeat.

After training:

```text
Initial loss: 3.297271
Final loss:   1.946110
```

The reduction in loss shows that the weight matrix has learned the character-transition patterns found in the training corpus.

Generation remains autoregressive: the sampled character becomes the context for the following prediction.

Open the folder:

[`v02-numpy-neural-model`](v02-numpy-neural-model/)

## Project versions

| Version | Main concept | Status |
|---|---|---|
| Version 1 | Character statistical model | Completed |
| Version 2 | Neural character model with NumPy | Completed |
| Version 3 | Training foundations | Planned |
| Version 4 | TensorFlow introduction | Planned |
| Version 5 | Embeddings and context windows | Planned |
| Version 6 | Recurrent language model | Planned |
| Version 7 | GRU or LSTM model | Planned |
| Version 8 | Attention | Planned |
| Version 9 | Transformer block | Planned |
| Version 10 | Mini decoder-only language model | Planned |
| Version 11 | Subword tokenizer and public datasets | Planned |
| Version 12 | Kaggle training pipeline | Planned |
| Version 13 | Evaluation and controlled generation | Planned |
| Version 14 | Instruction tuning | Planned |
| Version 15 | Advanced small LLM | Planned |

## Included checks

The Version 2 notebook verifies:

- the correct number of adjacent-character examples;
- the numerical input and target representations;
- the shape of the one-hot input matrix;
- the shape of the trainable weight matrix;
- that the probability distributions sum to 1;
- that the final loss is lower than the initial loss;
- that the recorded loss decreases during training;
- reproducible generation with the same seed;
- the requested output length.

The notebook can be executed from top to bottom without a GPU. NumPy is the only external computational dependency.

## Documentation

The repository includes a complete general project report in Italian and English:

- [Project report - English](project-report-en.pdf)
- [Relazione del progetto - Italiano](project-report-it.pdf)

Each completed version folder contains:

- a Jupyter Notebook;
- an Italian technical report;
- an English technical report;
- a version-specific architecture diagram.

The repository also includes a general project roadmap infographic:

- [Building LLM roadmap infographic](infographic.png)

## Repository structure

```text
building-llm/
├── v01-character-model/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 1 - Modello Statistico a Caratteri.pdf
│   ├── Report Version 1 - Character Statistical Model.pdf
│   └── Version 1.png
│
├── v02-numpy-neural-model/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 2 - Modello Neurale a Caratteri con NumPy.pdf
│   ├── Report Version 2 - Neural Character Model with NumPy.pdf
│   └── Version 2.png
│
├── infographic.png
├── project-report-en.pdf
├── project-report-it.pdf
├── README.md
├── LICENSE
└── .gitignore
```

## Run locally

### Requirements

- Python 3
- NumPy
- Jupyter Notebook or JupyterLab

Version 1 uses only Python's standard library. Version 2 introduces NumPy for numerical representation and model training. No GPU or machine-learning framework is required.

Clone the repository:

```bash
git clone https://github.com/lucalullo/building-llm.git
cd building-llm
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
v02-numpy-neural-model/building-llm.ipynb
```

Run the cells in order.

## Development approach

The project follows one main principle:

> **One version, one concept, one verifiable result.**

Instead of starting with a complex Transformer, the project introduces each mechanism through small and understandable steps.

Each new version continues the same educational journey while preserving the completed stages as reference implementations.

## Roadmap

- [x] Character statistical language model
- [x] Neural character language model with NumPy
- [ ] Training objective, optimization and data splits
- [ ] TensorFlow and automatic differentiation
- [ ] Embeddings and larger context windows
- [ ] Recurrent neural networks
- [ ] GRU or LSTM
- [ ] Scaled dot-product attention
- [ ] Transformer block
- [ ] Decoder-only Transformer
- [ ] Subword tokenization and public datasets
- [ ] Kaggle training pipeline and checkpoints
- [ ] Evaluation and controlled generation
- [ ] Instruction tuning
- [ ] Reproducible advanced small LLM

## Current limitations

Version 2 is intentionally simple:

- the context contains only one character;
- the training corpus is small and embedded in the notebook;
- training and inspection use the same examples;
- all examples are processed together in one full batch;
- there is no hidden layer or embedding;
- gradients are calculated manually;
- the model does not use automatic differentiation or a machine-learning framework;
- generated text shows local patterns but no long-term coherence.

These limitations define the current stage of the project.

## License

This project is distributed under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
