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

Version 4 introduces TensorFlow and automatic differentiation while preserving the same character-level neural language model and training workflow implemented in Version 3.

It continues directly from Version 3:

- the same English training corpus is used;
- adjacent characters remain the input-target examples;
- characters are converted into numerical identifiers;
- input identifiers are represented as TensorFlow one-hot tensors;
- the trainable weight matrix is stored in a `tf.Variable`;
- the dataset is divided into training and validation sets;
- the training examples are shuffled reproducibly;
- gradient descent is performed using mini-batches;
- training runs across multiple epochs;
- `tf.GradientTape` calculates gradients automatically;
- training and validation loss are monitored separately;
- text is generated one character at a time using a local random seed.

The model still contains 729 trainable parameters. The dataset contains 439 adjacent-character examples, divided into 351 training examples and 88 validation examples.

Training runs for 100 epochs using mini-batches of 32 examples. The training loss decreases from approximately `3.2960` to `1.9875`, while the best validation loss is approximately `2.8934` and is reached at epoch 64.

The first versions are language models, but they are not yet large language models. The LLM label becomes appropriate later in the project, after the introduction of subword tokenization, a decoder-only Transformer, a meaningful parameter count and training on a substantial corpus.

## Current architecture

```text
English Training Text
          ↓
Adjacent Character Pairs
          ↓
Numerical Character IDs
          ↓
TensorFlow One-Hot Tensors
          ↓
Train / Validation Split
          ↓
Training Data Shuffle
          ↓
Mini-Batches
          ↓
Trainable tf.Variable Weight Matrix
          ↓
Logits and Softmax Probabilities
          ↓
Cross-Entropy Loss
          ↓
Automatic Gradients with tf.GradientTape
          ↓
Gradient Descent
          ↓
Training / Validation Monitoring
          ↓
Next-Character Sampling
          ↓
Generated Text
```

The model still uses one character as context. The neural architecture remains unchanged, while TensorFlow tensors, trainable variables and automatic differentiation replace the corresponding manual NumPy calculations.

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

## Version 3 - Training Foundations

Version 3 preserves the same character-level neural model introduced in Version 2 and focuses on the foundations of a more structured training procedure.

The model still uses the same trainable weight matrix with shape:

```text
(27, 27)
```

This means that the architecture still contains:

```text
27 × 27 = 729 trainable parameters
```

The complete dataset contains 439 adjacent-character examples, which are divided reproducibly into:

```text
Training examples:   351
Validation examples: 88
```

Training is performed using mini-batches:

```text
Batch size: 32
Epochs:     100
```

The training procedure follows this sequence:

1. divide the examples into training and validation sets;
2. shuffle the training examples reproducibly;
3. divide the training examples into mini-batches;
4. calculate the logits;
5. convert the logits into probabilities with softmax;
6. calculate the cross-entropy loss;
7. calculate the gradients;
8. update the weights with gradient descent;
9. repeat across multiple epochs;
10. evaluate training and validation loss separately.

After training:

```text
Initial training loss:   3.2972
Final training loss:     1.9876
Initial validation loss: 3.2975
Final validation loss:   2.9037
Best validation loss:    2.8938
Best validation epoch:   64
```

The training loss continues to decrease, while the validation loss reaches its minimum around epoch 64 and then increases slightly.

Version 3 monitors this behavior but intentionally does not yet implement early stopping or automatic restoration of the best model parameters.

Generation remains autoregressive: the sampled character becomes the context for the following prediction.

Open the folder:

[`v03-training-foundations`](v03-training-foundations/)

## Version 4 - Introduction to TensorFlow

Version 4 rebuilds the same neural character language model with TensorFlow without changing its architecture.

The model uses TensorFlow operations for its main calculations:

- `tf.one_hot` creates the input tensors;
- `tf.Variable` stores the trainable weight matrix;
- `tf.matmul` calculates the logits;
- `tf.nn.softmax` converts logits into probabilities;
- `tf.nn.sparse_softmax_cross_entropy_with_logits` calculates the loss;
- `tf.GradientTape` calculates gradients automatically;
- `assign_sub` updates the weights.

The model still uses a weight matrix with shape:

```text
(27, 27)
```

This means that the architecture still contains:

```text
27 × 27 = 729 trainable parameters
```

The training configuration remains:

```text
Training examples:   351
Validation examples: 88
Batch size:          32
Epochs:              100
```

After training:

```text
Initial training loss:   3.2960
Final training loss:     1.9875
Initial validation loss: 3.2974
Final validation loss:   2.9035
Best validation loss:    2.8934
Best validation epoch:   64
```

The results remain very close to Version 3. Small differences are expected because TensorFlow uses `float32` tensors and a different random number generator.

The tests also verify that the gradients calculated automatically by TensorFlow match the gradients calculated manually.

Generation remains autoregressive and uses the final weights from epoch 100.

Open the folder:

[`v04-tensorflow-introduction`](v04-tensorflow-introduction/)

## Project versions

| Version | Main concept | Status |
|---|---|---|
| Version 1 | Character statistical model | Completed |
| Version 2 | Neural character model with NumPy | Completed |
| Version 3 | Training foundations | Completed |
| Version 4 | TensorFlow introduction | Completed |
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

The Version 4 notebook verifies:

- the correct number of adjacent-character examples;
- the numerical input and target representations;
- that the inputs are TensorFlow tensors;
- that the weights are trainable TensorFlow variables;
- the shape of the one-hot input tensor;
- the shape of the trainable weight matrix;
- that the training and validation sets contain the expected number of examples;
- that training and validation indices do not overlap;
- that the probability distributions sum to 1;
- that the final training loss is lower than the initial training loss;
- that the recorded training and validation losses remain finite;
- that automatic and manual gradients match;
- reproducible training with the same seed;
- reproducible generation with the same seed;
- the requested output length.

The notebook can be executed from top to bottom without a GPU. NumPy and TensorFlow are the external computational dependencies.

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
├── v03-training-foundations/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 3 - Fondamenti del Training.pdf
│   ├── Report Version 3 - Training Foundations.pdf
│   └── Version 3.png
│
├── v04-tensorflow-introduction/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 4 - Introduzione a TensorFlow.pdf
│   ├── Report Version 4 - Introduction to TensorFlow.pdf
│   └── Version 4.png
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
- TensorFlow
- Jupyter Notebook or JupyterLab

Version 1 uses only Python's standard library. Versions 2 and 3 use NumPy for numerical representation and model training. Version 4 uses NumPy for reproducible shuffling and sampling, while TensorFlow performs the model calculations and training. No GPU is required.

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
v04-tensorflow-introduction/building-llm.ipynb
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
- [x] Training objective, optimization and data splits
- [x] TensorFlow and automatic differentiation
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

Version 4 is intentionally simple:

- the context contains only one character;
- the training corpus is small and embedded in the notebook;
- the model still contains only one trainable weight matrix;
- there is no hidden layer or embedding;
- the model remains a character-level neural bigram model;
- TensorFlow automates the gradients but does not change the architecture;
- the train/validation split is performed on individual adjacent-character examples;
- validation loss is monitored, but early stopping is not implemented;
- the best model parameters are not automatically restored;
- generated text shows local patterns but no long-term coherence.

These limitations define the current stage of the project.

## License

This project is distributed under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
