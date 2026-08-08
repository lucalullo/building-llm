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

Version 8 replaces the gated recurrent processing from Version 7 with explicit scaled dot-product attention.

It continues directly from Version 7:

- the same 440-character English training corpus is used;
- the same 27-character vocabulary is preserved;
- each training example still uses four previous characters to predict the next character;
- each character identifier still selects an 8-dimensional trainable embedding vector;
- recurrent processing is removed;
- a trainable positional embedding is added at each of the four context positions so that sequence order remains explicit;
- the positioned embeddings are projected into query, key and value representations;
- queries and keys are compared with scaled dot products;
- softmax converts the scaled scores into normalized attention weights;
- the attention weights combine the value vectors into contextual representations;
- only the contextual representation of the final position is used for next-character prediction;
- `tf.GradientTape` calculates gradients for all six trainable parameter matrices;
- the parameters are updated manually with gradient descent;
- training and validation loss are monitored separately;
- text generation remains autoregressive and uses a sliding four-character context window.

The model contains 1064 trainable parameters: 216 values in the character embedding matrix, 32 in the positional embedding matrix, 128 values in each of the query, key and value projection matrices, and 432 in the output matrix. The dataset still contains 436 context-target examples, divided into 348 training examples and 88 validation examples.

Training runs for 100 epochs using mini-batches of 32 examples. The training loss decreases from approximately `3.2958` to `2.7354`. Validation loss reaches its best value of approximately `2.8849` at epoch 74 and finishes at approximately `2.9271` at epoch 100, suggesting mild overfitting after the best validation point.

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
Trainable Character Embedding Matrix (27, 8)
         +
Trainable Positional Embedding Matrix (4, 8)
         ↓
Positioned Context Representations (4 × 8)
         ↓
Query / Key / Value Projections (4 × 16)
         ↓
Scaled Dot-Product Scores QKᵀ / √dₖ
         ↓
Softmax Attention Weights (4 × 4)
         ↓
Weighted Value Combinations
         ↓
Contextual Representations (4 × 16)
         ↓
Final Contextual Representation (16 values)
         ↓
Trainable Output Weight Matrix (16, 27)
         ↓
Logits and Softmax Probabilities
         ↓
Cross-Entropy Loss
         ↓
Automatic Gradients with tf.GradientTape
         ↓
Manual Gradient Descent
         ↓
Training / Validation Monitoring
         ↓
Sliding-Window Next-Character Sampling
         ↓
Generated Text
```

The model still uses four previous characters as context. Version 8 removes recurrent processing and adds positional information explicitly. The positioned embeddings are projected into queries, keys and values, scaled dot-product attention lets the four context positions exchange information directly, and only the contextual representation of the final position is passed to the output projection for next-character prediction.
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

## Version 6 - Recurrent Language Model

Version 6 replaces the flattened context representation from Version 5 with a simple recurrent neural network.

The same four-character context windows and trainable embeddings are preserved, but the four embeddings are now processed sequentially from left to right.

At every context position, the model updates a hidden state using the current embedding and the previous hidden state:

```text
current embedding + previous hidden state
                    ↓
                   tanh
                    ↓
             new hidden state
```

The recurrent model uses the following trainable matrices:

```text
Embedding matrix:        (27, 8)
Input-to-hidden weights:  (8, 16)
Recurrent weights:       (16, 16)
Output weights:          (16, 27)
```

The complete architecture therefore contains:

```text
27 × 8 + 8 × 16 + 16 × 16 + 16 × 27 = 1032 trainable parameters
```

The dataset remains unchanged from Version 5:

```text
Training examples:   348
Validation examples: 88
```

The training configuration remains intentionally simple:

```text
Learning rate: 1.0
Batch size:    32
Epochs:        100
Seed:          42
```

After training:

```text
Initial training loss:   3.2958
Final training loss:     1.5251
Initial validation loss: 3.2958
Final validation loss:   3.0352
Best validation loss:    2.8396
Best validation epoch:   92
```

The recurrent model learns slowly during the first part of training. Validation loss reaches its minimum much later than in Version 5 and increases only slightly afterward, showing the beginning of overfitting.

Generation remains autoregressive with a sliding four-character context window. Each window is processed recurrently from a zero hidden state, matching the training procedure.

Open the folder:

[`v06-recurrent-language-model`](v06-recurrent-language-model/)

## Version 7 - GRU Language Model

Version 7 replaces the simple recurrent hidden-state update from Version 6 with a gated recurrent unit (GRU).

The same four-character context windows, trainable embeddings and 16-dimensional hidden state are preserved. The four embeddings are still processed sequentially from left to right, but the recurrent update now uses learned gates.

At every context position, the model calculates:

```text
update gate
reset gate
candidate hidden state
        ↓
gated hidden-state update
```

The update gate controls how much of the previous hidden state is preserved. The reset gate controls how much previous information contributes to the candidate state. The candidate state uses `tanh`, while the update and reset gates use sigmoid activations.

The GRU uses the following trainable matrices:

```text
Embedding matrix:             (27, 8)
Update input weights:          (8, 16)
Update recurrent weights:     (16, 16)
Reset input weights:           (8, 16)
Reset recurrent weights:      (16, 16)
Candidate input weights:       (8, 16)
Candidate recurrent weights:  (16, 16)
Output weights:               (16, 27)
```

The complete architecture therefore contains:

```text
27 × 8
+ 3 × (8 × 16 + 16 × 16)
+ 16 × 27
= 1800 trainable parameters
```

The dataset remains unchanged from Version 6:

```text
Training examples:   348
Validation examples: 88
```

The training configuration remains intentionally simple:

```text
Learning rate: 1.0
Batch size:    32
Epochs:        100
Seed:          42
```

After training:

```text
Initial training loss:   3.2958
Final training loss:     2.6596
Initial validation loss: 3.2958
Final validation loss:   2.9139
Best validation loss:    2.8995
Best validation epoch:   99
```

The GRU learns slowly during the first part of training. Validation loss reaches its minimum at epoch 99 and increases slightly at the final epoch, suggesting the beginning of mild overfitting.

Generation remains autoregressive with a sliding four-character context window. Each window is processed by the GRU from a zero hidden state, matching the training procedure.

Open the folder:

[`v07-gru-language-model`](v07-gru-language-model/)

## Version 8 - Attention

Version 8 replaces the gated recurrent processing from Version 7 with explicit scaled dot-product attention.

The same four-character context windows and trainable character embeddings are preserved, but recurrence is removed. Because the recurrent model previously represented sequence order implicitly, Version 8 adds trainable positional embeddings to the four context positions.

The positioned context is projected into queries, keys and values:

```text
Q = X Wq
K = X Wk
V = X Wv
```

Queries and keys produce scaled attention scores:

```text
scores = Q Kᵀ / √dk
```

Softmax converts the scores into normalized attention weights:

```text
attention = softmax(scores)
```

The attention weights combine the value vectors:

```text
H = attention V
```

For the four-character context, the model uses the following trainable matrices:

```text
Character embedding matrix:  (27, 8)
Positional embedding matrix:   (4, 8)
Query projection:             (8, 16)
Key projection:               (8, 16)
Value projection:             (8, 16)
Output weights:              (16, 27)
```

The complete architecture therefore contains:

```text
27 × 8
+ 4 × 8
+ 3 × (8 × 16)
+ 16 × 27
= 1064 trainable parameters
```

The dataset remains unchanged from Version 7:

```text
Training examples:   348
Validation examples: 88
```

The training configuration remains intentionally simple:

```text
Learning rate: 1.0
Batch size:    32
Epochs:        100
Seed:          42
```

After training:

```text
Initial training loss:   3.2958
Final training loss:     2.7354
Initial validation loss: 3.2958
Final validation loss:   2.9271
Best validation loss:    2.8849
Best validation epoch:   74
```

The model changes very little during the first part of training, then the loss decreases more clearly. Validation loss reaches its minimum at epoch 74. After that point, training loss continues to decrease while validation loss increases slightly, suggesting mild overfitting.

Attention produces a `4 × 4` matrix for each example. The rows are normalized by softmax and each row sums to 1. For next-character prediction, only the contextual representation of the final position is passed to the output layer.

Generation remains autoregressive with a sliding four-character context window. For every new window, character embeddings and positional embeddings are combined, attention is recomputed, and the final contextual representation produces the next-character probabilities.

Open the folder:

[`v08-attention`](v08-attention/)

## Project versions

| Version | Main concept | Status |
|---|---|---|
| Version 1 | Character statistical model | Completed |
| Version 2 | Neural character model with NumPy | Completed |
| Version 3 | Training foundations | Completed |
| Version 4 | TensorFlow introduction | Completed |
| Version 5 | Embeddings and context window | Completed |
| Version 6 | Recurrent language model | Completed |
| Version 7 | GRU language model | Completed |
| Version 8 | Attention | Completed |
| Version 9 | Transformer block | Planned |
| Version 10 | Mini decoder-only language model | Planned |
| Version 11 | Subword tokenizer and public datasets | Planned |
| Version 12 | Kaggle training pipeline | Planned |
| Version 13 | Evaluation and controlled generation | Planned |
| Version 14 | Instruction tuning | Planned |
| Version 15 | Advanced small LLM | Planned |

## Included checks

The Version 8 notebook verifies:

- the correct number of four-character context-target examples;
- the numerical input and target representations;
- the construction of the first sliding context windows directly from the corpus;
- that the inputs are TensorFlow tensors;
- that all six trainable parameter matrices are TensorFlow variables;
- the shapes of the input tensor, character embedding matrix, positional embedding matrix, query, key and value projection matrices, attention output and logits;
- that the model contains exactly 1064 trainable parameters;
- that the attention matrix has shape `4 × 4` for each example;
- that attention weights remain finite and between 0 and 1;
- that every row of attention weights sums to 1;
- that the training and validation sets together cover all examples;
- that training and validation indices do not overlap;
- that training and validation loss histories contain the expected number of measurements;
- that the final training and validation losses are lower than their initial values;
- that the recorded training and validation losses remain finite;
- that next-character probability distributions sum to 1;
- that TensorFlow produces finite gradients for all six trainable parameter matrices;
- that the gradient shapes match the corresponding variables;
- that every trainable parameter matrix actually changes during training;
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
├── v06-recurrent-language-model/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 6 - Modello Linguistico Ricorrente.pdf
│   ├── Report Version 6 - Recurrent Language Model.pdf
│   └── Version 6.png
│
├── v07-gru-language-model/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 7 - Modello Linguistico GRU.pdf
│   ├── Report Version 7 - GRU Language Model.pdf
│   └── Version 7.png
│
│
├── v08-attention/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 8 - Attention.pdf
│   ├── Report Version 8 - Attention.pdf
│   └── Version 8.png
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

Version 1 uses only Python's standard library. Versions 2 and 3 use NumPy for numerical representation and model training. Versions 4, 5, 6, 7 and 8 use NumPy for reproducible data handling and sampling, while TensorFlow performs the model calculations, trainable parameter updates and automatic differentiation. No GPU is required.

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
v08-attention/building-llm.ipynb
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
- [x] Recurrent neural networks
- [x] GRU
- [x] Scaled dot-product attention
- [ ] Transformer block
- [ ] Decoder-only Transformer
- [ ] Subword tokenization and public datasets
- [ ] Kaggle training pipeline and checkpoints
- [ ] Evaluation and controlled generation
- [ ] Instruction tuning
- [ ] Reproducible advanced small LLM

## Current limitations

Version 8 is intentionally simple:

- the context still has a fixed length of four characters;
- the training corpus is small and embedded directly in the notebook;
- neighboring context windows overlap and the random example-level split can place similar windows in both training and validation;
- the model uses one explicit scaled dot-product attention computation;
- only the contextual representation of the final position contributes directly to next-character prediction loss;
- the model intentionally does not use bias terms;
- causal masking is not used yet because the target character remains outside the input context;
- residual connections are not used yet;
- normalization is not used yet;
- a feed-forward network is not used yet;
- multi-head attention is not used yet;
- training uses plain mini-batch gradient descent with a fixed learning rate;
- validation loss is monitored, but early stopping is not implemented;
- the best model parameters are not automatically restored;
- generation uses the final epoch-100 parameters even though validation loss is best at epoch 74;
- validation is not a robust estimate of generalization to a different corpus;
- generated text captures local character patterns but still has no long-term coherence.

These limitations define the current stage of the project and prepare the transition to a Transformer block in Version 9.
## License

This project is distributed under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
