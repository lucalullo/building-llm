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

Version 5 extends the TensorFlow neural character language model from Version 4 with learned embeddings and a fixed four-character context window.

It continues directly from Version 4:

- the same 440-character English training corpus is used;
- the same 27-character vocabulary is preserved;
- each training example now uses four previous characters to predict the next character;
- character identifiers replace one-hot vectors as the direct model inputs;
- each character identifier selects an 8-dimensional trainable embedding vector;
- the four embeddings are concatenated into a 32-value context representation;
- a trainable output weight matrix converts the context representation into next-character logits;
- the dataset is divided reproducibly into training and validation sets;
- the training examples are shuffled and processed in mini-batches;
- `tf.GradientTape` calculates gradients for both trainable parameter sets;
- training and validation loss are monitored separately;
- text generation remains autoregressive and now uses a sliding four-character context window.

The model contains 1080 trainable parameters: 216 values in the embedding matrix and 864 values in the output weight matrix. The dataset contains 436 context-target examples, divided into 348 training examples and 88 validation examples.

Training runs for 100 epochs using mini-batches of 32 examples. The training loss decreases from approximately `3.2958` to `0.5043`. Validation loss reaches its best value of approximately `2.8413` at epoch 14 and then increases to approximately `6.6350` by epoch 100, showing clear overfitting on the small corpus.

The first versions are language models, but they are not yet large language models. The LLM label becomes appropriate later in the project, after the introduction of subword tokenization, a decoder-only Transformer, a meaningful parameter count and training on a substantial corpus.

## Current architecture

```text
English Training Text
          ↓
4-Character Context Windows
          ↓
Numerical Character IDs
          ↓
Train / Validation Split
          ↓
Training Data Shuffle
          ↓
Mini-Batches
          ↓
Trainable Embedding Matrix (27, 8)
          ↓
4 Embeddings per Context
          ↓
Flattened Context Vector (32 values)
          ↓
Trainable Output Weight Matrix (32, 27)
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
Sliding-Window Next-Character Sampling
          ↓
Generated Text
```

The model now uses four previous characters as context instead of one. Learned embeddings replace sparse one-hot vectors, but the four embeddings are still concatenated into a fixed flattened representation and sent directly through a linear output projection.

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

## Version 5 - Embeddings and Context Window

Version 5 changes the predictive model for the first time since the neural bigram was introduced.

Instead of representing one current character with a one-hot vector, the model now uses a fixed context containing four character identifiers:

```text
mode → l
odel → i
deli → n
elin → g
```

Each identifier selects one row from a trainable embedding matrix:

```text
Embedding matrix: (27, 8)
```

The four selected embedding vectors are concatenated:

```text
4 × 8 = 32 context values
```

The flattened context is multiplied by the trainable output matrix:

```text
Output weight matrix: (32, 27)
```

The complete architecture therefore contains:

```text
27 × 8 + 32 × 27 = 1080 trainable parameters
```

The dataset contains 436 context-target examples, divided reproducibly into:

```text
Training examples:   348
Validation examples: 88
```

The training configuration remains intentionally close to Version 4:

```text
Learning rate: 1.0
Batch size:    32
Epochs:        100
Seed:          42
```

After training:

```text
Initial training loss:   3.2958
Final training loss:     0.5043
Initial validation loss: 3.2958
Final validation loss:   6.6350
Best validation loss:    2.8413
Best validation epoch:   14
```

Training loss decreases strongly, while validation loss reaches its minimum at epoch 14 and then increases substantially. This shows clear overfitting: the larger context-based model fits the small training corpus much more strongly than the earlier one-character model.

Generation remains autoregressive, but the context now moves as a sliding four-character window. After a new character is sampled, the oldest context character is removed and the new character becomes part of the next prediction.

Open the folder:

[`v05-embeddings-context-window`](v05-embeddings-context-window/)

## Project versions

| Version | Main concept | Status |
|---|---|---|
| Version 1 | Character statistical model | Completed |
| Version 2 | Neural character model with NumPy | Completed |
| Version 3 | Training foundations | Completed |
| Version 4 | TensorFlow introduction | Completed |
| Version 5 | Embeddings and context window | Completed |
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

The Version 5 notebook verifies:

- the correct number of four-character context-target examples;
- the numerical input and target representations;
- the construction of the first sliding context windows directly from the corpus;
- that the inputs are TensorFlow tensors;
- that the embedding matrix and output weights are trainable TensorFlow variables;
- the shapes of the input tensor, embedding matrix and output weight matrix;
- that the training and validation sets together cover all examples;
- that training and validation indices do not overlap;
- that training and validation loss histories contain the expected number of measurements;
- that the final training loss is lower than the initial training loss;
- that the recorded training and validation losses remain finite;
- that the probability distributions sum to 1;
- that TensorFlow produces finite gradients for both trainable parameter sets;
- that the gradient shapes match the corresponding variables;
- that both the embedding matrix and output weights actually change during training;
- reproducible retraining from the same initial parameters and seed;
- reproducible generation with the same seed;
- the requested output length and starting context.

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
├── v05-embeddings-context-window/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 5 - Embedding e Finestra di contesto.pdf
│   ├── Report Version 5 - Embeddings and Context Window.pdf
│   └── Version 5.png
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

Version 1 uses only Python's standard library. Versions 2 and 3 use NumPy for numerical representation and model training. Versions 4 and 5 use NumPy for reproducible data handling and sampling, while TensorFlow performs the model calculations, trainable parameter updates and automatic differentiation. No GPU is required.

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
v05-embeddings-context-window/building-llm.ipynb
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
- [x] Embeddings and larger context windows
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

Version 5 is intentionally simple:

- the context has a fixed length of four characters;
- the training corpus is small and embedded directly in the notebook;
- neighboring context windows overlap and the random example-level split can place similar windows in both training and validation;
- the model uses a trainable embedding matrix and one trainable output weight matrix, but no hidden nonlinear layer;
- the four embeddings are concatenated into a fixed flattened representation;
- there is no recurrent state or attention mechanism;
- validation loss is monitored, but early stopping is not implemented;
- the best model parameters are not automatically restored;
- generation uses the final epoch-100 parameters even though validation loss is best at epoch 14;
- validation is not a robust estimate of generalization to a different corpus;
- generated text captures stronger local character patterns but still has no long-term coherence.

These limitations define the current stage of the project and prepare the transition to sequence modeling in Version 6.

## License

This project is distributed under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
