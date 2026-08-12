# Building LLM

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

A progressive educational project that shows how language models are built step by step, from a simple character-level statistical model to a small decoder-only Transformer with public data, subword tokenization, checkpointed training, held-out evaluation, controlled generation, instruction tuning and explicit multi-head causal self-attention.

The project begins with standard Python and introduces one new concept in each version. The goal is not to compete with industrial models, but to make tokenization, probability, training, neural networks, attention and autoregressive generation easier to understand.

## Project roadmap

The complete project evolves through five main stages:

1. statistical foundations;
2. neural foundations;
3. sequence models;
4. attention and Transformer architecture;
5. a small decoder-only language model with public data, checkpointed training, evaluation, controlled generation, instruction tuning and multi-head causal self-attention.

Each stage introduces the concepts required by the next one while keeping the implementation understandable and verifiable.

![Building LLM project roadmap infographic](infographic.png)

## Current version

Version 15 is the final planned stage of the original project roadmap. It preserves the complete Version 14 pipeline and introduces one isolated architectural change: **explicit multi-head causal self-attention**.

It continues directly from Version 14:

- WikiText-103 Raw v1 remains the public language-modeling dataset;
- the BPE tokenizer is still trained only on the selected WikiText training subset;
- the 2048-token vocabulary and 16-token context remain unchanged;
- training, validation and test split boundaries remain separate;
- the two-block decoder-only Transformer is preserved;
- the model still contains exactly 23 trainable TensorFlow variables and 34,496 trainable parameters;
- base training still uses the Version 12 LAST/BEST checkpoint pipeline;
- held-out WikiText evaluation from Version 13 is preserved;
- temperature, top-k and top-p decoding controls remain available;
- the deterministic 256-row Databricks Dolly 15K subset from Version 14 is preserved;
- the instruction split remains 192 training / 32 validation / 32 test rows;
- prompt formatting remains `Instruction / optional Context / Response`;
- supervised instruction loss still uses response-focused masking;
- the BEST instruction checkpoint is still selected only from masked validation loss;
- base-versus-tuned controlled generation remains part of the final experiment;
- the 16-dimensional attention projection is now split into **2 independent heads of 8 dimensions each**;
- each head applies its own causal scaled dot-product attention;
- every head is normalized independently and cannot attend to future positions;
- the two head outputs are concatenated back to 16 dimensions;
- the same `16 × 8` attention-output projection returns to the 8-dimensional model representation;
- no additional trainable matrices are introduced, so the parameter count remains unchanged.

The key Version 15 attention shapes are:

```text
Input representation:      (batch, 16, 8)
Projected Q / K / V:       (batch, 16, 16)
Split into heads:          (batch, 2, 16, 8)
Per-head attention:        (batch, 2, 16, 16)
Concatenated head output:  (batch, 16, 16)
Attention output:          (batch, 16, 8)
```

The final model configuration remains:

```text
Vocabulary size:          2048
Context length:             16
Embedding dimension:         8
Attention dimension:        16
Attention heads:             2
Head dimension:              8
Feed-forward dimension:     16
Transformer blocks:          2
Trainable variables:        23
Trainable parameters:   34,496
```

Base language-model training still uses:

```text
Learning rate: 1.0
Batch size:    32
Epochs:        20
Seed:          42
```

The saved Version 15 base-model execution produces:

```text
Initial training loss:      7.626327
Initial validation loss:    7.625874
Best validation epoch:             18
Best validation loss:       5.075184
Final training loss:        4.982092
Final validation loss:      5.075184

Held-out cross-entropy:      5.098042
Held-out perplexity:       163.701
Top-1 accuracy:              0.0986
Top-5 accuracy:              0.2401
Uniform CE baseline:         7.624619
Uniform PPL baseline:     2048.0
```

Instruction tuning uses:

```text
Dataset:                  Databricks Dolly 15K
Selected rows:            256
Train / validation / test: 192 / 32 / 32
Instruction batch size:    32
Instruction learning rate: 0.15
Instruction epochs:         8
Max useful windows/example: 2
Generated BPE tokens:      16
```

The saved Version 15 instruction-tuning execution produces:

```text
Initial masked train loss:       5.273911
Initial masked validation loss:  5.297913
Best instruction epoch:                 5
Best instruction validation loss: 5.180870
Final masked train loss:         4.883737
Final masked validation loss:    5.180870
Instruction test masked loss:    5.182242
Instruction test perplexity:   178.082
```

These values describe the saved execution. Small numerical differences may appear across TensorFlow environments.

Version 15 completes the original V1-V15 educational arc without turning the final notebook into a production training framework. The final architectural change remains small, explicit, inspectable and directly testable.

## Current architecture

```text
WikiText-103 Raw v1
        ↓
Official Train / Validation / Test Splits
        ↓
Remove Empty Rows + Independent Seed-Controlled Subsets
        ↓
BPE Tokenizer
(trained only on selected WikiText training text)
        ↓
2048-Token Vocabulary
        ↓
17-Token Windows
        ↓
Input: 16 BPE Tokens
Target: Same Window Shifted by 1 Token
        ↓
12k / 2k / 2k Capped Train / Validation / Test Sequences
        ↓
Trainable Token Embeddings (2048, 8)
        +
Trainable Positional Embeddings (16, 8)
        ↓
Positioned Representations (16 × 8)
        ↓
Transformer Block 1
        ↓
Q / K / V Projections (8 → 16)
        ↓
Split into 2 Heads × 8 Dimensions
        ↓
Independent Causal Scaled Dot-Product Attention per Head
        ↓
Per-Head Attention Matrices (16 × 16)
        ↓
Concatenate Heads (16D)
        ↓
Attention Output Projection (16 → 8)
        ↓
Residual + LayerNorm
        ↓
Feed-Forward Network (8 → 16 → 8)
        ↓
Residual + LayerNorm
        ↓
Transformer Block 2
        ↓
Same Explicit 2-Head Causal Attention Pipeline
        ↓
Vocabulary Logits at All 16 Positions
        ↓
Sparse Cross-Entropy Loss
        ↓
Manual Mini-Batch Gradient Descent with tf.GradientTape
        ↓
Train Epochs 1–10
        ↓
LAST Checkpoint
        ↓
Fresh Model Restore + Resume Epochs 11–20
        ↓
BEST Checkpoint Selected by Validation Loss
        ↓
Fresh BEST-Restored Base Model
        ↓
Held-Out WikiText Evaluation
Cross-Entropy / Perplexity / Top-1 / Top-5
        ↓
Controlled Generation
Greedy / Temperature / Top-k / Top-p
        ↓
Databricks Dolly 15K Subset
192 Train / 32 Validation / 32 Test
        ↓
Instruction / Optional Context / Response Formatting
        ↓
16-Token Response-Focused Windows
        ↓
Response Mask
0 = prompt target / 1 = response target
        ↓
Masked Supervised Fine-Tuning
8 Epochs / LR 0.15
        ↓
BEST Instruction Checkpoint
        ↓
Held-Out Dolly Evaluation
        ↓
Base BEST vs Tuned BEST Controlled Generation
        ↓
Per-Head Attention Inspection + Final Tests
```

The important Version 15 distinction is internal to self-attention. Versions 11 through 14 use one 16-dimensional attention space per Transformer block. Version 15 reorganizes the same 16 projected dimensions into two independently normalized 8-dimensional causal heads, then concatenates them before the unchanged output projection. Because the trainable matrix shapes are preserved, the model remains exactly 34,496 parameters.

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

[`v05-embeddings-context`](v05-embeddings-context/)

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
Final training loss:     1.892438
Initial validation loss: 3.291214
Final validation loss:   2.220907
Best validation loss:    2.141364
Best validation epoch:   68
```

For context `mode`, the saved final-position probabilities begin with:

```text
space  0.4307
t      0.1878
l      0.1060
r      0.0732
c      0.0668
s      0.0394
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

[`v11-subword-tokenization-and-public-dataset`](v11-subword-tokenization-and-public-dataset/)


## Version 12 - Training Pipeline and Checkpoints

Version 12 preserves the complete Version 11 WikiText/BPE language-model experiment and adds explicit training-state management.

The model architecture, tokenizer, selected dataset rows, sequence caps, context length, parameter count, learning rate and total training budget remain unchanged.

The new concept is the separation between:

```text
LAST → where training can resume
BEST → which model validation selects
FINAL → fresh model restored from BEST for test and analysis
```

Training is intentionally interrupted after epoch 10. A fresh model restores the LAST checkpoint and resumes from epoch 11 through epoch 20.

The checkpoint state remains deliberately minimal:

```text
LAST:
model
completed_epoch
best_epoch
best_validation_loss

BEST:
model
best_epoch
best_validation_loss
```

The runtime directory is recreated at the beginning of each Run All. Persistent cross-session checkpoint infrastructure is intentionally outside the scope of this educational version.

The saved execution produces:

```text
Initial training loss:    7.626330
Initial validation loss:  7.625952
Best validation epoch:          18
Best validation loss:     5.074273
Final training loss:      4.986566
Final validation loss:    5.074273
Test loss:                5.105811
Uniform baseline:         7.624619
```

The important lesson is that LAST and BEST are not necessarily the same model. LAST answers where training ended; BEST answers which observed parameters validation selected.

Open the folder:

[`v12-training-pipeline-and-checkpoints`](v12-training-pipeline-and-checkpoints/)

## Version 13 - Evaluation and Controlled Generation

Version 13 keeps the complete Version 12 training and checkpoint pipeline unchanged and adds a clearer held-out evaluation stage together with explicit decoding controls.

The validation-selected BEST model is restored before the WikiText test split is measured.

The new evaluation reports:

```text
Cross-entropy
Perplexity
Top-1 token accuracy
Top-5 token accuracy
```

The saved held-out execution produces:

```text
Cross-entropy:  5.105321
Perplexity:   164.897
Top-1 accuracy: 0.1030
Top-5 accuracy: 0.2402
Evaluated tokens: 32,000
```

Version 13 also implements decoding controls directly from logits:

```text
Greedy:          argmax
Temperature:     0.7 / 1.3
Top-k:           k = 20
Top-p:           p = 0.9
```

All generation modes:

- start from the same held-out prompt;
- keep the same 16-token context;
- add exactly 40 new BPE tokens;
- use a local NumPy random generator for stochastic sampling;
- leave model weights unchanged.

The tests verify the expected relationships and invariants, including `perplexity = exp(cross-entropy)`, top-5 accuracy being at least top-1 accuracy, finite normalized distributions, exactly 20 non-zero top-k probabilities, reproducible seeded top-p sampling and preserved causal attention.

Open the folder:

[`v13-evaluation-and-controlled-generation`](v13-evaluation-and-controlled-generation/)

## Version 14 - Instruction Tuning

Version 14 preserves the complete Version 13 base model and adds a second learning stage: **supervised instruction tuning**.

A deterministic subset of **Databricks Dolly 15K** is used:

```text
Selected rows: 256

Training:       192
Validation:      32
Test:            32
```

Each record is converted into an explicit prompt:

```text
Instruction: {instruction}

Context: {context}        # only when non-empty

Response: {response}
```

The BPE tokenizer from WikiText is reused rather than retrained on Dolly.

The key new mechanism is response-focused masking. Prompt tokens remain visible to causal attention, but only response targets contribute to the supervised objective:

```text
prompt target   → response_mask = 0
response target → response_mask = 1
```

The masked objective is:

```text
masked_loss =
sum(token_loss × response_mask)
/
sum(response_mask)
```

Instruction tuning uses:

```text
Batch size:       32
Learning rate:  0.15
Epochs:             8
Shuffle seed: SEED + 1000 + epoch
```

The instruction model is created as a fresh instance and initialized exactly from the selected base BEST parameters.

Dolly validation selects a separate BEST instruction checkpoint. The held-out Dolly test split is measured only after selection.

The saved instruction execution produces:

```text
Initial masked training loss:    5.281624
Initial masked validation loss:  5.333013
Best instruction epoch:                 8
Best instruction validation loss: 5.191426
Final masked training loss:      4.788731
Final masked validation loss:    5.191427
Instruction test masked loss:    5.231103
Instruction test perplexity:   186.999
```

Base and instruction-tuned models are compared on the same Dolly test prompt with greedy and top-p `0.9` generation, adding 16 BPE tokens.

This comparison demonstrates the mechanics of supervised adaptation. The notebook deliberately does **not** claim a guaranteed semantic improvement and does not compute instruction-adherence, helpfulness, safety or human-preference scores.

Open the folder:

[`v14-instruction-tuning`](v14-instruction-tuning/)

## Version 15 - Explicit Multi-Head Causal Self-Attention

Version 15 preserves the complete Version 14 data, training, evaluation, checkpoint and instruction-tuning pipeline. The only new concept is explicit **multi-head causal self-attention**.

The existing 16-dimensional attention projection is reorganized into:

```text
Number of heads: 2
Head dimension:  8
Total attention dimension: 16
```

For each Transformer block:

```text
Input:                  (batch, 16, 8)
Q / K / V:              (batch, 16, 16)
Split:                  (batch, 2, 16, 8)
Attention scores:       (batch, 2, 16, 16)
Per-head causal softmax
Head outputs:           (batch, 2, 16, 8)
Concatenate:            (batch, 16, 16)
Output projection:      (batch, 16, 8)
```

Each head computes its own scaled dot-product scores, causal mask, softmax distribution and weighted value combination.

The tests verify independently for every head that:

- attention rows sum to approximately 1;
- all future-position entries are zero;
- the attention tensor has the expected `(1, 2, 16, 16)` shape for the inspected example.

The trainable matrices remain the same shapes as Version 14:

```text
Q projection:          8 × 16
K projection:          8 × 16
V projection:          8 × 16
Attention output:     16 × 8
```

Therefore Version 15 introduces **no additional trainable matrices** and the complete model remains:

```text
23 trainable TensorFlow variables
34,496 trainable parameters
```

The saved Version 15 base held-out evaluation produces:

```text
Cross-entropy:  5.098042
Perplexity:   163.701
Top-1 accuracy: 0.0986
Top-5 accuracy: 0.2401
```

The saved instruction stage produces:

```text
Best instruction epoch:                 5
Best instruction validation loss: 5.180870
Final masked training loss:      4.883737
Final masked validation loss:    5.180870
Instruction test masked loss:    5.182242
Instruction test perplexity:   178.082
```

Version 15 closes the planned V1-V15 arc with a final architectural evolution that is isolated, inspectable and testable without increasing model size.

Open the folder:

[`v15-multi-head-transformer-capstone`](v15-multi-head-transformer-capstone/)

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
| Version 10 | Decoder-only Transformer | Completed |
| Version 11 | Subword tokenization and public dataset | Completed |
| Version 12 | Training pipeline and checkpoints | Completed |
| Version 13 | Evaluation and controlled generation | Completed |
| Version 14 | Instruction tuning | Completed |
| Version 15 | Explicit multi-head causal self-attention | Completed |

## Included checks

Every version ends with an explicit test section and the completed notebooks finish with:

```text
All checks passed.
```

The final Version 15 notebook verifies the complete accumulated pipeline, including:

- tokenizer vocabulary and configured tensor shapes;
- the 2048-token BPE vocabulary and fixed 16-token contexts;
- shifted next-token target construction;
- exactly two independent Transformer blocks;
- exactly 23 trainable TensorFlow variables;
- exactly 34,496 trainable parameters;
- LAST checkpoint restoration after the simulated interruption;
- resume from epoch 11 through epoch 20;
- BEST checkpoint selection using validation loss only;
- held-out WikiText evaluation only after BEST restoration;
- finite cross-entropy and perplexity values;
- `perplexity = exp(cross-entropy)`;
- top-5 token accuracy greater than or equal to top-1 accuracy;
- finite, non-negative and normalized controlled probability distributions;
- exact top-k support size;
- reproducible seeded top-p sampling;
- causal attention normalization;
- zero attention to future positions;
- exactly two attention heads;
- head dimension equal to 8;
- expected per-head attention shape `(1, 2, 16, 16)` for the inspected example;
- independent causal masking in every attention head;
- disjoint Dolly train / validation / test source rows;
- explicit `Instruction / optional Context / Response` formatting;
- binary response masks;
- at least one response target in every retained instruction window;
- instruction-model initialization from base BEST;
- parameter updates during supervised fine-tuning;
- BEST instruction-checkpoint selection and restoration;
- finite masked training, validation and test losses;
- reproducible controlled instruction generation;
- expected generation lengths for both WikiText and Dolly demonstrations.

The notebooks are designed to execute from top to bottom without requiring a GPU.

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
├── v11-subword-tokenization-and-public-dataset/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 11 - Tokenizzazione Subword e Dataset Pubblico.pdf
│   ├── Report Version 11 - Subword Tokenization and Public Dataset.pdf
│   └── Version 11.png
│
├── v12-training-pipeline-and-checkpoints/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 12 - Pipeline di Training e Checkpoint.pdf
│   ├── Report Version 12 - Training Pipeline and Checkpoints.pdf
│   └── Version 12.png
│
├── v13-evaluation-and-controlled-generation/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 13 - Valutazione e Generazione Controllata.pdf
│   ├── Report Version 13 - Evaluation and Controlled Generation.pdf
│   └── Version 13.png
│
├── v14-instruction-tuning/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 14 - Instruction Tuning.pdf
│   ├── Report Version 14 - Instruction Tuning.pdf
│   └── Version 14.png
│
├── v15-multi-head-transformer-capstone/
│   ├── building-llm.ipynb
│   ├── Relazione Versione 15 - Self-Attention Causale Multi-Head Esplicita.pdf
│   ├── Report Version 15 - Explicit Multi-Head Causal Self-Attention.pdf
│   └── Version 15.png
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

Versions 11 through 15 additionally use Pandas to read WikiText-103 Parquet files and the Hugging Face `tokenizers` library to train and apply the BPE tokenizer.

Versions 14 and 15 also require the external Databricks Dolly 15K instruction dataset file expected by the notebook.

No GPU is required.

Versions 11 through 15 require the documented WikiText-103 Raw v1 dataset files to be available at the dataset path expected by the notebook. The saved notebooks are configured for the attached Kaggle dataset input.

Versions 14 and 15 deliberately do not download or silently replace missing Dolly data. If the external instruction dataset is unavailable, the notebook stops explicitly.

Clone the repository:

```bash
git clone https://github.com/lucalullo/building-llm.git
cd building-llm
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

To inspect the final planned version, open:

```text
v15-multi-head-transformer-capstone/building-llm.ipynb
```

Run the cells in order.

## Development approach

The project follows one main principle:

> **One version, one concept, one verifiable result.**

Instead of starting with a complex Transformer, the project introduces each mechanism through small and understandable steps.

Each new version continues the same educational journey while preserving completed stages as reference implementations.

Version 9 introduces the internal structure of one explicit causal Transformer block.

Version 10 reorganizes the same mathematics into reusable components and stacks two independent Transformer blocks.

Version 11 deliberately keeps the Transformer core unchanged and moves the educational focus to data representation: public WikiText data and BPE subword tokenization replace the tiny hardcoded character corpus.

Version 12 keeps the model unchanged and makes training state explicit through LAST and BEST checkpoints, restore and resume.

Version 13 keeps training unchanged and separates model selection, held-out test evaluation and decoding strategy.

Version 14 keeps the base architecture unchanged and adds supervised instruction tuning through explicit prompt formatting and response-focused masked loss.

Version 15 keeps the full data, training, evaluation and instruction pipeline unchanged and introduces explicit two-head causal self-attention without increasing the parameter count.

This separation keeps architectural changes, data changes, training-state changes, evaluation changes and supervised adaptation directly inspectable instead of introducing them all at once.

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
- [x] Training pipeline and checkpoints
- [x] Evaluation and controlled generation
- [x] Instruction tuning
- [x] Explicit multi-head causal self-attention

The original V1-V15 roadmap is complete.

## Current limitations

Version 15 is intentionally small, explicit and educational:

- only controlled subsets of WikiText-103 Raw v1 are used rather than full-dataset training;
- the language-model experiment uses 10,000 training rows, 1,000 validation rows and 1,000 test rows;
- the model uses at most 12,000 training sequences, 2,000 validation sequences and 2,000 test sequences;
- context length remains fixed at 16 BPE tokens;
- the tokenizer vocabulary contains only 2048 tokens;
- the whitespace-based BPE configuration is intentionally simple;
- whitespace pre-tokenization can produce spaces before punctuation after decoding;
- the model representation dimension remains only 8;
- total attention dimension remains 16;
- multi-head attention uses only two 8-dimensional heads;
- the feed-forward hidden dimension remains 16;
- the model contains only two Transformer blocks;
- no dropout is used;
- attention, feed-forward and output projections intentionally omit bias terms;
- base training still uses manual mini-batch gradient descent with fixed learning rate `1.0`;
- no learning-rate schedule, Adam, gradient clipping, mixed precision or large-scale distributed training is introduced;
- the checkpoint state is deliberately minimal and the runtime directory is recreated for each clean Run All;
- persistent resume across unrelated Kaggle sessions is outside the project scope;
- controlled generation uses simple greedy, temperature, top-k and top-p rules rather than advanced search or production serving;
- token-level WikiText metrics do not measure factuality, usefulness or instruction quality;
- the Dolly experiment uses only 256 selected instruction records;
- the inherited 16-token context truncates long instruction prompts;
- the BPE tokenizer is reused rather than retrained on Dolly;
- instruction tuning uses response-only supervised loss;
- no preference optimization, RLHF, DPO, PPO, tool use or agent framework is included;
- base-versus-tuned generation is a mechanical comparison and does not claim guaranteed semantic improvement;
- the result remains a small educational language model rather than a practical large language model.

These limitations are intentional. The project was designed to expose mechanisms clearly rather than maximize model quality or reproduce production-scale LLM engineering.

## Project status

**Building LLM is complete according to the V1-V15 structure originally planned for the project.**

The repository now contains the full educational path from statistical character counts to a small decoder-only Transformer with public subword data, explicit checkpointed training, held-out evaluation, controlled generation, supervised instruction tuning and explicit multi-head causal self-attention.

Version 15 therefore closes the original roadmap rather than opening another required development phase.

The project can remain finished in this form. Documentation, compatibility fixes or small maintenance updates may still be made when useful.

At the same time, the repository is intentionally not declared permanently frozen. If a future concept is worth studying with the same **one version, one concept, one verifiable result** philosophy, Building LLM may one day receive additional versions beyond Version 15.

There is currently no required Version 16: any future continuation would be a new extension of an already completed project.

## License

This project is distributed under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
