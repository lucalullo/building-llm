# Building LLM

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

A progressive educational project that shows how language models are built step by step, from a simple character-level statistical model to a small decoder-only Transformer using subword tokenization and a documented public dataset.

The project begins with standard Python and introduces one new concept in each version. The goal is not to compete with industrial models, but to make tokenization, probability, training, neural networks, attention and autoregressive generation easier to understand.

## Project roadmap

The complete project evolves through five main stages:

1. statistical foundations;
2. neural foundations;
3. sequence models;
4. attention and Transformer architecture;
5. a small decoder-only language model with subword tokenization and public data.

Each stage introduces the concepts required by the next one while keeping the implementation understandable and verifiable.

![Building LLM project roadmap infographic](infographic.png)

## Current version

Version 11 keeps the small decoder-only Transformer introduced in Version 10 and changes how language enters the model.

It continues directly from Version 10:

- the two reusable causal Transformer blocks are preserved;
- residual connections, LayerNorm operations and position-wise feed-forward networks remain unchanged;
- the tiny hardcoded 440-character corpus is replaced by the public **WikiText-103 Raw v1** dataset;
- the original training, validation and test split boundaries are preserved;
- controlled subsets are sampled independently from the three official splits;
- character-level tokenization is replaced by **Byte Pair Encoding (BPE)**;
- the BPE tokenizer is trained only on the selected training subset;
- the tokenizer uses a vocabulary of 2048 tokens, including the special `[UNK]` token;
- each WikiText row is tokenized independently;
- fixed-length windows contain 17 BPE tokens;
- the first 16 tokens form the model input and the same sequence shifted by one token forms the 16-token target;
- a stride of 16 limits overlap and incomplete final windows are discarded;
- token embeddings replace character embeddings;
- trainable positional embeddings now represent 16 sequence positions;
- both Transformer blocks preserve an 8-dimensional model representation;
- vocabulary logits are produced at all 16 sequence positions;
- sparse softmax cross-entropy is calculated over all sequential targets;
- `tf.GradientTape` calculates gradients for all 23 trainable TensorFlow variables;
- sparse embedding gradients are updated with `scatter_sub`, while dense gradients use `assign_sub`;
- training, validation and test losses are evaluated separately;
- next-token probabilities are inspected on unseen test data;
- generation uses deterministic greedy decoding with a sliding 16-token BPE context;
- causal attention is inspected independently in both Transformer blocks.

The original WikiText-103 Raw v1 split sizes are:

```text
Training rows:   1,801,350
Validation rows:     3,760
Test rows:           4,358
```

Version 11 uses controlled subsets:

```text
Selected training rows:   10,000
Selected validation rows:  1,000
Selected test rows:        1,000

Used training sequences:   12,000
Used validation sequences:  2,000
Used test sequences:        2,000
```

The tokenizer vocabulary and model configuration are:

```text
Vocabulary size:     2048
Context length:        16
Embedding dimension:    8
Attention dimension:   16
Feed-forward dimension: 16
Transformer blocks:     2
```

The model contains exactly 34,496 trainable parameters:

```text
Token embeddings:              2048 × 8  = 16,384
Positional embeddings:           16 × 8  =    128
Transformer block 1:                         800
Transformer block 2:                         800
Output weights:                   8 × 2048 = 16,384
--------------------------------------------------
Total:                                     34,496
```

Training uses:

```text
Learning rate: 1.0
Batch size:    32
Epochs:        20
Seed:          42
```

The saved execution produces:

```text
Initial training loss:   7.626767
Final training loss:     4.987922
Initial validation loss: 7.626187
Final validation loss:   5.085637
Test loss:               5.117153
Uniform baseline:        7.624619
```

These values describe the saved execution. Small numerical differences may appear across TensorFlow environments.

Version 11 completes the transition from a tiny character-level educational model to a small decoder-only Transformer that operates on learned subword tokens from a documented public dataset. The Transformer core remains intentionally small and inspectable so the effect of the new data representation can be studied directly.

## Current architecture

```text
WikiText-103 Raw v1
        ↓
Original Train / Validation / Test Splits
        ↓
Remove Empty Text Rows
        ↓
Independent Seed-Controlled Subsets
        ↓
BPE Tokenizer
(trained only on selected training text)
        ↓
2048-Token Vocabulary
        ↓
Each Text Row Tokenized Independently
        ↓
17-Token Fixed-Length Windows
        ↓
Input:  16 BPE Tokens
Target: Same Window Shifted by 1 Token
        ↓
Stride = 16
        ↓
Capped Train / Validation / Test Sequences
        ↓
Trainable Token Embedding Matrix (2048, 8)
        +
Trainable Positional Embedding Matrix (16, 8)
        ↓
Positioned Token Representations (16 × 8)
        ↓
Transformer Block 1
        ↓
Query / Key / Value Projections (16 × 16)
        ↓
Causal Scaled Dot-Product Attention (16 × 16)
        ↓
Attention Output Projection (16 → 8)
        ↓
Residual + LayerNorm
        ↓
Position-Wise Feed-Forward Network (8 → 16 → 8)
        ↓
Residual + LayerNorm
        ↓
Transformer Block 2
        ↓
Query / Key / Value Projections (16 × 16)
        ↓
Causal Scaled Dot-Product Attention (16 × 16)
        ↓
Attention Output Projection (16 → 8)
        ↓
Residual + LayerNorm
        ↓
Position-Wise Feed-Forward Network (8 → 16 → 8)
        ↓
Residual + LayerNorm
        ↓
Final Transformer Output (16 × 8)
        ↓
Trainable Output Weight Matrix (8, 2048)
        ↓
Vocabulary Logits at All 16 Positions
        ↓
Sparse Cross-Entropy Loss over All 16 Targets
        ↓
Automatic Gradients with tf.GradientTape
        ↓
Manual Mini-Batch Gradient Descent
        ↓
Training / Validation Monitoring
        ↓
Final Test Evaluation
        ↓
Final-Position Next-Token Probabilities
        ↓
Sliding 16-Token Greedy Autoregressive Generation
        ↓
Generated Text
```

The important architectural distinction from Version 10 is at the input and output boundaries: character identifiers and a 27-character vocabulary are replaced by BPE token identifiers and a 2048-token vocabulary. The internal two-block causal Transformer structure remains deliberately unchanged.

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
Query projection:              (8, 16)
Key projection:                (8, 16)
Value projection:              (8, 16)
Output weights:               (16, 27)
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

## Version 9 - Transformer Block

Version 9 transforms the explicit attention mechanism from Version 8 into a complete single causal Transformer block.

The four-character context and trainable character and positional embeddings are preserved, but the training objective changes from one next-character target per context to four aligned sequential targets.

For example:

```text
Input:  mode
Target: odel
```

The positioned context is projected into queries, keys and values:

```text
Q = X Wq
K = X Wk
V = X Wv
```

Scaled dot-product scores are calculated:

```text
scores = Q Kᵀ / √dk
```

A causal mask prevents every sequence position from seeing future positions:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

The masked scores are normalized with softmax:

```text
attention = softmax(masked_scores)
```

The attention weights combine the value vectors:

```text
H = attention V
```

The attention output is projected from 16 dimensions back to the 8-dimensional model representation:

```text
16 → 8
```

The first Transformer sublayer then applies:

```text
positioned input
       +
attention output
       ↓
Residual
       ↓
LayerNorm
```

A position-wise feed-forward network follows:

```text
8 → 16 → 8
```

The feed-forward output is combined with another residual connection and LayerNorm:

```text
normalized attention representation
                +
       feed-forward output
                ↓
             Residual
                ↓
             LayerNorm
```

Vocabulary logits are produced independently at all four sequence positions:

```text
Transformer output: (batch, 4, 8)
        ↓
Output weights:     (8, 27)
        ↓
Logits:             (batch, 4, 27)
```

Cross-entropy loss is calculated over all four aligned next-character targets.

The complete model contains 13 trainable TensorFlow variables and exactly:

```text
1264 trainable parameters
```

The dataset contains:

```text
Total examples:      436
Training examples:   348
Validation examples: 88
Batch size:           32
Updates per epoch:    11
```

The training configuration is:

```text
Learning rate: 1.0
Batch size:    32
Epochs:        100
Seed:          42
```

The saved training execution produces:

```text
Initial training loss:   3.298020
Final training loss:     1.915883
Initial validation loss: 3.299714
Final validation loss:   2.208963
Best validation loss:    2.130791
Best validation epoch:   67
```

For the context `mode`, the saved final-position probabilities begin with:

```text
space  0.3648
r      0.1843
d      0.0762
l      0.0639
t      0.0583
m      0.0465
```

The saved causal attention matrix for `mode` is:

```text
       m       o       d       e

m   1.0000  0.0000  0.0000  0.0000
o   0.9234  0.0766  0.0000  0.0000
d   0.2832  0.3961  0.3207  0.0000
e   0.1037  0.4917  0.2196  0.1849
```

Every row sums to approximately 1 and every value above the main diagonal is zero, confirming that future information is blocked.

Generation remains autoregressive. At every step the entire Transformer block is recomputed for the current four-character context, but only the logits from the final position are used to sample the next character.

The saved generated text begins:

```text
moderetr win mawiom n.
sinttl macors mod manes lexper fconged clerning lamamp belers,
th comakerincte crsmay rin tleleaserks pamake codvame win mexpl...
```

Version 9 is the first stage of the project with the main internal structure of a Transformer block:

```text
embeddings + positions
        ↓
Q / K / V
        ↓
causal self-attention
        ↓
output projection
        ↓
residual + LayerNorm
        ↓
feed-forward network
        ↓
residual + LayerNorm
        ↓
vocabulary logits
```

Open the folder:

[`v09-transformer-block`](v09-transformer-block/)

## Version 10 - Decoder-Only Transformer

Version 10 turns the explicit single Transformer block from Version 9 into a small decoder-only Transformer composed of two reusable, independent Transformer blocks.

The same 440-character English corpus, 27-character vocabulary, four-character context, sequential next-character targets and character-level training objective are preserved.

The main change is architectural and organizational:

- the Transformer operations are grouped into a reusable `TransformerBlock`;
- two independent block instances are stacked sequentially;
- the output of Transformer block 1 becomes the input of Transformer block 2;
- both blocks preserve the `(batch, 4, 8)` representation shape;
- each block has its own query, key, value, attention-output, LayerNorm and feed-forward parameters;
- causal masking is applied independently inside both blocks;
- model parameters are grouped instead of being passed individually through the training code;
- the internal Transformer mathematics remains explicit and inspectable.

The complete flow is:

```text
Characters
    ↓
Integer IDs
    ↓
4-Character Context + 4 Sequential Targets
    ↓
Character + Positional Embeddings
    ↓
Transformer Block 1
    ↓
Transformer Block 2
    ↓
Vocabulary Logits at All 4 Positions
    ↓
Cross-Entropy Loss
    ↓
Autoregressive Generation
```

The model contains 23 trainable TensorFlow variables and exactly:

```text
Character embeddings:           27 × 8  = 216
Positional embeddings:           4 × 8  = 32
Transformer block 1:                      800
Transformer block 2:                      800
Output weights:                  8 × 27 = 216
---------------------------------------------
Total:                                  2064
```

Each Transformer block contains exactly 800 parameters:

```text
Query projection:               8 × 16 = 128
Key projection:                 8 × 16 = 128
Value projection:               8 × 16 = 128
Attention output projection:   16 × 8  = 128
LayerNorm 1 scale + shift:      8 + 8   = 16
Feed-forward input:             8 × 16 = 128
Feed-forward output:           16 × 8  = 128
LayerNorm 2 scale + shift:      8 + 8   = 16
--------------------------------------------
Per block:                               800
```

The dataset remains:

```text
Total examples:      436
Training examples:   348
Validation examples: 88
Batch size:           32
Updates per epoch:    11
```

The saved training execution produces:

```text
Initial training loss:   3.295300
Final training loss:     1.844497
Initial validation loss: 3.291214
Final validation loss:   2.177593
Best validation loss:    2.126416
Best validation epoch:   98
```

For context `mode`, the saved final-position probabilities begin with:

```text
space  0.3901
t      0.1992
l      0.1369
r      0.0717
c      0.0697
s      0.0373
```

Both blocks preserve the causal constraint. Every attention row sums to approximately 1 and every future-position weight is zero.

Generation remains autoregressive with a sliding four-character window. At every step, the complete context passes through both Transformer blocks and only the final-position logits are sampled.

Version 10 is the first project stage where the Transformer is organized as a reusable decoder-only stack rather than a single explicit block.

Open the folder:

[`v10-decoder-only-transformer`](v10-decoder-only-transformer/)

## Version 11 - Subword Tokenization and Public Dataset

Version 11 keeps the two-block decoder-only Transformer from Version 10 and changes the representation of language entering the model.

The tiny hardcoded character corpus is replaced by **WikiText-103 Raw v1**, while character identifiers are replaced by BPE subword token identifiers.

The original dataset split boundaries are preserved:

```text
Training rows:   1,801,350
Validation rows:     3,760
Test rows:           4,358
```

After removing empty rows, controlled subsets are sampled independently from each split:

```text
Training rows used:   10,000
Validation rows used:  1,000
Test rows used:        1,000
```

The BPE tokenizer:

- is trained only on the selected training subset;
- uses a maximum vocabulary size of 2048 tokens;
- includes a special `[UNK]` token;
- uses the `</w>` suffix to mark the end of a pre-tokenized word;
- never uses validation or test text to build the vocabulary.

Each WikiText row is tokenized independently.

Token windows contain:

```text
CONTEXT_LENGTH + 1 = 17 tokens
```

The first 16 tokens form the input and the shifted 16-token sequence forms the target:

```text
Input:  [t0, t1, t2, ..., t15]
Target: [t1, t2, t3, ..., t16]
```

A stride of 16 limits overlap, incomplete final windows are discarded, and examples never cross text-row boundaries.

Before sequence caps, the saved preprocessing produces:

```text
Training windows:   96,893
Validation windows:  9,764
Test windows:        9,572
```

The model uses:

```text
Training sequences:   12,000
Validation sequences:  2,000
Test sequences:        2,000

Vocabulary size:       2048
Context length:          16
Embedding dimension:      8
Attention dimension:     16
Feed-forward dimension:  16
Transformer blocks:       2
```

The model contains 23 trainable TensorFlow variables and exactly:

```text
Token embeddings:              2048 × 8  = 16,384
Positional embeddings:           16 × 8  =    128
Transformer block 1:                         800
Transformer block 2:                         800
Output weights:                   8 × 2048 = 16,384
--------------------------------------------------
Total:                                     34,496
```

Training uses sparse softmax cross-entropy at all 16 sequence positions. `tf.GradientTape` calculates gradients for the 23 trainable variables. Sparse `IndexedSlices` gradients from token embedding lookups are updated with `scatter_sub`, while dense gradients use `assign_sub`.

The training configuration is:

```text
Learning rate: 1.0
Batch size:    32
Epochs:        20
Seed:          42
```

The saved execution produces:

```text
Initial training loss:   7.626767
Final training loss:     4.987922
Initial validation loss: 7.626187
Final validation loss:   5.085637
Test loss:               5.117153
Uniform baseline:        7.624619
```

For the unseen test context:

```text
Little Gidding is the fourth and final poem
```

the highest saved final-position probabilities begin with:

```text
,      0.1392
.      0.0825
and    0.0471
of     0.0453
'      0.0336
(      0.0253
was    0.0252
```

The true next token is:

```text
of</w>
```

with probability `0.0453`.

The first Transformer block has an almost uniform final-position attention distribution, with weights close to `0.0625`. The second block is more selective; its largest saved final-position weights include:

```text
G      0.2545
L      0.1621
f      0.0861
the    0.0600
```

Every attention row sums to approximately 1 and future-position attention remains zero.

Generation uses deterministic greedy decoding. Starting from:

```text
Little Gidding is the fourth and final poem
```

the saved 40-token demonstration produces:

```text
Little Gidding is the fourth and final poem , and the Sea , and the Sea , and the Sea
, and the Sea , and the Sea , and the Sea , and the S
```

Version 11 completes the transition from character-level educational language modeling to a public-data subword pipeline while deliberately preserving the small two-block decoder-only Transformer architecture from Version 10.

Open the folder:

[`v11-subword-tokenization-public-dataset`](v11-subword-tokenization-public-dataset/)

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
| Version 9 | Transformer block | Completed |
| Version 10 | Mini decoder-only language model | Completed |
| Version 11 | Subword tokenization and public datasets | Completed |
| Version 12 | Training pipeline and checkpoints | Planned |
| Version 13 | Evaluation and controlled generation | Planned |
| Version 14 | Instruction tuning | Planned |
| Version 15 | Advanced small LLM | Planned |

## Included checks

The Version 11 notebook verifies:

- that the tokenizer vocabulary size is exactly 2048;
- that training, validation and test input/target tensors use the configured 16-token sequence length and capped split sizes;
- that token identifiers remain inside the tokenizer vocabulary;
- that the first target sequence is the corresponding input shifted by one token;
- that the model output has shape `(1, 16, 8)`;
- that vocabulary logits have shape `(1, 16, 2048)`;
- that the model contains exactly two independent Transformer block objects;
- that the model contains exactly 23 trainable TensorFlow variables;
- that the complete model contains exactly 34,496 trainable parameters;
- that both attention matrices have shape `(16, 16)`;
- that every attention row sums to approximately 1;
- that every future-position attention weight is zero;
- that final training loss is lower than initial training loss;
- that final validation loss is lower than initial validation loss;
- that test loss remains below the uniform-vocabulary baseline;
- that training and validation loss histories contain exactly 20 finite values;
- that final training, validation and test losses are finite;
- that the inspected next-token probability distribution is finite and sums to approximately 1;
- that final training and validation losses remain below the uniform-vocabulary baseline;
- that the two Transformer block instances are distinct objects;
- that autoregressive generation extends the prompt;
- that autoregressive generation adds exactly 40 new BPE tokens.

The final notebook test section finishes with:

```text
All checks passed.
```

The notebook can be executed from top to bottom without a GPU. NumPy, Pandas, TensorFlow and the Hugging Face `tokenizers` library are the main external computational dependencies used by Version 11.

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
├── v05-embeddings-context/
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
├── v08-attention/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 8 - Attention.pdf
│   ├── Report Version 8 - Attention.pdf
│   └── Version 8.png
│
├── v09-transformer-block/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 9 - Transformer Block.pdf
│   ├── Report Version 9 - Transformer Block.pdf
│   └── Version 9.png
│
├── v10-decoder-only-transformer/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 10 - Decoder Only Transformer.pdf
│   ├── Report Version 10 - Decoder Only Transformer.pdf
│   └── Version 10.png
│
├── v11-subword-tokenization-public-dataset/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 11 - Tokenizzazione Subword e Dataset Pubblico.pdf
│   ├── Report Version 11 - Subword Tokenization and Public Dataset.pdf
│   └── Version 11.png
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
- Pandas
- TensorFlow
- Hugging Face `tokenizers`
- Jupyter Notebook or JupyterLab

Version 1 uses only Python's standard library.

Versions 2 and 3 use NumPy for numerical representation and model training.

Versions 4 through 10 use NumPy and TensorFlow for numerical operations, model calculations, trainable parameter updates and automatic differentiation.

Version 11 additionally uses Pandas to read WikiText-103 Parquet files and the Hugging Face `tokenizers` library to train and apply the BPE tokenizer.

No GPU is required.

Version 11 also requires the documented WikiText-103 Raw v1 dataset files to be available at the dataset path expected by the notebook. The saved notebook is configured for the attached Kaggle dataset input.

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
v11-subword-tokenization-public-dataset/building-llm.ipynb
```

Run the cells in order.

## Development approach

The project follows one main principle:

> **One version, one concept, one verifiable result.**

Instead of starting with a complex Transformer, the project introduces each mechanism through small and understandable steps.

Each new version continues the same educational journey while preserving the completed stages as reference implementations.

Version 9 introduces the internal structure of one explicit causal Transformer block.

Version 10 reorganizes that same mathematics into reusable components and stacks two independent Transformer blocks, making the decoder-only architecture easier to extend without hiding its internal operations.

Version 11 deliberately keeps the Transformer core unchanged and moves the educational focus to data representation: a documented public corpus, official dataset splits and BPE subword tokenization replace the tiny hardcoded character corpus.

This keeps the architectural transition and the data/tokenization transition separate and directly inspectable.

## Roadmap

- [x] Character statistical language model
- [x] Neural character language model with NumPy
- [x] Training objective, optimization and data splits
- [x] TensorFlow and automatic differentiation
- [x] Embeddings and larger context windows
- [x] Recurrent neural networks
- [x] GRU
- [x] Scaled dot-product attention
- [x] Transformer block
- [x] Decoder-only Transformer
- [x] Subword tokenization and public datasets
- [ ] Training pipeline and checkpoints
- [ ] Evaluation and controlled generation
- [ ] Instruction tuning
- [ ] Reproducible advanced small LLM

## Current limitations

Version 11 is intentionally small and explicit:

- only controlled subsets of WikiText-103 Raw v1 are used rather than full-dataset training;
- the selected experiment uses 10,000 training rows, 1,000 validation rows and 1,000 test rows;
- the model uses at most 12,000 training sequences, 2,000 validation sequences and 2,000 test sequences;
- the context length is fixed at 16 BPE tokens;
- the tokenizer vocabulary contains only 2048 tokens;
- the BPE configuration uses simple whitespace pre-tokenization;
- whitespace pre-tokenization can produce spaces before punctuation after decoding;
- the model representation dimension remains only 8;
- the attention dimension is only 16;
- the feed-forward hidden dimension is only 16;
- the model contains only two Transformer blocks;
- each block uses one attention mechanism rather than multi-head attention;
- dropout is not used;
- attention, feed-forward and output projections intentionally do not use bias terms;
- training uses plain mini-batch gradient descent with a fixed learning rate of `1.0`;
- no learning-rate schedule is used;
- no mixed-precision training is used;
- checkpoint saving, restoration and resume logic are not yet implemented;
- the model retains the final epoch parameters rather than automatically restoring a best validation checkpoint;
- generation uses deterministic greedy decoding rather than temperature, top-k or top-p sampling;
- greedy generation can amplify repetitive output patterns;
- evaluation remains intentionally lightweight;
- more systematic evaluation and controlled generation are reserved for later versions;
- the model is still a small educational language model rather than a practical large language model.

These limitations define the current stage of the project and prepare the transition to **Version 12: Training pipeline and checkpoints**.

## License

This project is distributed under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
