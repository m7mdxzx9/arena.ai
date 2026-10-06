from .schema import C, Q, N

DL = "Deep Learning"
NT, VL = "neural_tower", "vision_lab"

CONCEPTS = [
    C("neuron", "Neuron Intuition", DL, NT, ["vectors", "classification"],
      explain="An artificial neuron multiplies each input by a weight, adds them up with a bias, and passes the total through an activation "
              "function: output = activation(w·x + b). Alone, it is basically logistic regression.",
      analogy="A tiny judge weighing evidence: each clue gets an importance (weight), plus a default leaning (bias).",
      visual="neuron_playground",
      questions=[
          N(1, "Inputs [1, 2], weights [0.5, 1.0], bias 0, linear activation. Output?", 2.5, 0.001, "0.5·1 + 1.0·2 + 0 = 2.5."),
          Q(2, "A single sigmoid neuron is mathematically similar to…", ["k-means", "Logistic regression", "A decision tree", "PCA"], 1, "Linear score + sigmoid."),
          Q(3, "Why can't one neuron solve XOR?", ["Too slow", "Its decision boundary is a single straight line; XOR needs two", "XOR has no data", "It can"], 1,
            "Stacking neurons into layers creates curved / combined boundaries."),
      ], generators=["neuron_forward"]),

    C("weights", "Weights", DL, NT, ["neuron"],
      explain="Weights are the learned numbers that scale each input. Large positive weight: input pushes the output up strongly. "
              "Training = finding good weights. A network's 'knowledge' lives in its weights.",
      analogy="Volume knobs on a mixing desk, one per input channel.",
      visual="neuron_playground",
      questions=[
          Q(1, "Where is learned knowledge stored in a neural network?", ["In the dataset", "In the weights and biases", "In the activation function", "In the loss"], 1, "Parameters."),
          Q(2, "A weight of 0 means the corresponding input…", ["Dominates", "Has no effect on that neuron", "Is negative", "Is missing"], 1, "Multiplied away."),
          Q(3, "Why initialise weights randomly instead of all zeros?", ["Speed", "Identical weights make all neurons learn the same thing (symmetry)", "Zeros are illegal", "Saves memory"], 1,
            "Random initialisation breaks symmetry."),
      ]),

    C("bias", "Bias (parameter)", DL, NT, ["neuron"],
      explain="The bias is a learned constant added before the activation. It shifts the decision boundary away from the origin — the "
              "neuron's default tendency when inputs are zero. (Different from 'bias' in fairness!)",
      analogy="The intercept b in y = wx + b.",
      visual="neuron_playground",
      questions=[
          Q(1, "Without a bias, a line y = wx must pass through…", ["(1,1)", "The origin (0,0)", "Any point", "(0,1)"], 1, "The bias lets it shift."),
          N(2, "w=2, x=0, b=−1, linear. Output?", -1, 0.001, "2·0 − 1."),
          Q(3, "'Bias' in a neuron vs 'bias' in fairness are…", ["The same", "Different concepts that share a word", "Both hyperparameters", "Both metrics"], 1, "Context matters."),
      ]),

    C("activation", "Activation Functions", DL, NT, ["neuron", "math_functions"],
      explain="Activations add non-linearity. ReLU = max(0, z) (simple, popular); sigmoid squashes to (0,1); tanh to (−1,1). Without "
              "activations, stacking layers collapses into one linear function.",
      analogy="A bend in a pipe: without bends, every pipe network is just one straight pipe.",
      visual="activation_plot",
      questions=[
          N(1, "ReLU(−3)?", 0, 0.001, "max(0, −3) = 0."),
          Q(2, "A deep network with ONLY linear activations is equivalent to…", ["A deeper network", "A single linear model", "A tree", "A CNN"], 1, "Compositions of linear maps are linear."),
          Q(3, "Which activation suits the output layer for binary probability?", ["ReLU", "Sigmoid", "Linear", "None"], 1, "Output in (0,1)."),
      ], predict="activation_linear"),

    C("loss", "Loss Functions", DL, NT, ["mse", "probability"],
      explain="The loss is a single number measuring how wrong the model is on training data. Training minimises it. Regression often uses MSE; "
              "classification uses cross-entropy, which strongly punishes confident wrong answers.",
      analogy="The loss is the score in golf: lower is better.",
      questions=[
          Q(1, "During successful training, the training loss usually…", ["Increases", "Decreases", "Stays constant", "Becomes negative"], 1, "That's the point."),
          Q(2, "Cross-entropy is largest when the model is…", ["Confident and right", "Unsure", "Confident and wrong", "Untrained"], 2, "−log(small probability) is huge."),
          N(3, "Cross-entropy for true class probability 0.5 is −ln(0.5). Value? (2 dp)", 0.69, 0.01, "≈ 0.693."),
      ]),

    C("gradient_descent", "Gradient Descent", DL, NT, ["loss", "gradients"],
      explain="Repeat: compute loss → compute gradient of loss w.r.t. every weight → move each weight a small step opposite its gradient. "
              "The learning rate is the step size.",
      analogy="Walking downhill in fog by feeling the slope under your feet.",
      visual="gd_descent",
      questions=[
          Q(1, "Gradient descent updates weights using…", ["The test set", "The gradient of the loss", "Random numbers only", "Accuracy"], 1, "Slope information."),
          N(2, "w=1.0, gradient=−2, lr=0.1. New w?", 1.2, 0.001, "1.0 − 0.1·(−2) = 1.2."),
          Q(3, "Gradient descent finds…", ["Always the global minimum", "A minimum near its path (possibly local)", "The maximum", "Zero loss always"], 1, "Non-convex losses have many valleys."),
      ], generators=["gd_step"], predict="lr_explosion"),

    C("forward_pass", "Forward Pass", DL, NT, ["neuron", "matrices"],
      explain="The forward pass computes the prediction layer by layer: h = activation(W·x + b), then the next layer uses h as input, until the output.",
      analogy="Water flowing through a series of filters to the tap.",
      visual="nn_diagram",
      questions=[
          Q(1, "In the forward pass, data flows…", ["Output → input", "Input → output", "Randomly", "Only in training"], 1, "Forward."),
          Q(2, "Input 4 features, hidden layer 8 neurons. Weight matrix shape?", ["4×4", "4×8 (or 8×4)", "8×8", "1×8"], 1, "One weight per input-neuron pair."),
          N(3, "How many parameters in a dense layer from 4 inputs to 8 neurons (with biases)?", 40, 0.001, "4·8 weights + 8 biases."),
      ]),

    C("backprop", "Backpropagation (intuition)", DL, NT, ["forward_pass", "gradient_descent"],
      explain="Backpropagation efficiently computes every weight's gradient by applying the chain rule backwards from the loss through each layer. "
              "Each weight learns 'how much did I contribute to the error?'.",
      analogy="After a team loss, the coach traces back which passes led to the goal against — and how much each player contributed.",
      visual="nn_diagram",
      questions=[
          Q(1, "Backprop computes…", ["Predictions", "Gradients of the loss w.r.t. weights", "The dataset", "Accuracy"], 1, "Then gradient descent uses them."),
          Q(2, "Backprop relies mainly on which calculus rule?", ["Chain rule", "Product rule for integrals", "L'Hôpital", "None"], 0, "Derivatives of compositions."),
          Q(3, "With sigmoid activations in a very deep network, early layers may barely learn because…", ["Too much data", "Gradients shrink as they pass back (vanishing gradients)",
                                                                                                           "Biases", "Dropout"], 1, "One reason ReLU became popular."),
      ]),

    C("learning_rate", "Learning Rate", DL, NT, ["gradient_descent"],
      explain="The learning rate sets the step size. Too small → painfully slow. Too large → overshooting, oscillation, even divergence (loss → ∞/NaN). "
              "It is often the single most important hyperparameter.",
      analogy="Steering a car: tiny corrections take forever; huge jerks put you in the ditch.",
      visual="gd_descent",
      questions=[
          Q(1, "Loss explodes to NaN after a few steps. Most likely cause?", ["Learning rate too small", "Learning rate too large", "Too few epochs", "Good training"], 1, "Divergence."),
          Q(2, "Loss decreases, but extremely slowly over 500 epochs. Try…", ["Smaller LR", "Larger LR (carefully)", "Removing data", "Adding a test set"], 1, "Steps are too timid."),
          Q(3, "Adam with lr=0.001 vs SGD with lr=0.001 — same behaviour?", ["Yes", "No — optimizers scale steps differently, so good LRs differ", "Adam has no LR", "SGD has no LR"], 1, "LR depends on optimizer."),
      ], predict="lr_explosion"),

    C("epoch", "Epoch", DL, NT, ["gradient_descent"],
      explain="One epoch = one full pass over the training data. Training typically runs many epochs; too many can overfit (watch validation loss).",
      analogy="Reading the whole textbook once = one epoch.",
      questions=[
          N(1, "1,000 samples, batch size 100. Weight updates per epoch?", 10, 0.001, "1000/100."),
          Q(2, "Validation loss starts rising after epoch 30 while training loss keeps falling. Do what?", ["Train 300 more", "Stop early around epoch 30",
                                                                                                           "Increase LR", "Remove validation"], 1, "Early stopping."),
          Q(3, "More epochs always means better test performance.", ["True", "False"], 1, "Eventually it memorises."),
      ]),

    C("batch", "Batch / Mini-batch", DL, NT, ["epoch"],
      explain="Instead of computing the gradient on all data (slow) or one sample (noisy), we use mini-batches (e.g., 32 samples). "
              "Smaller batches = noisier but more frequent updates.",
      analogy="Tasting a spoonful of soup rather than drinking the whole pot before adjusting the salt.",
      questions=[
          Q(1, "Batch size 1 gives gradient estimates that are…", ["Exact", "Very noisy", "Zero", "Identical each time"], 1, "One sample is a rough estimate."),
          N(2, "600 samples, batch size 32. How many batches per epoch (last partial counts)?", 19, 0.001, "ceil(600/32) = 19."),
          Q(3, "Larger batches mostly improve…", ["Hardware throughput per step", "Noise for exploration", "Leakage", "Labels"], 0, "Parallelism on GPUs."),
      ]),

    C("optimizer", "Optimizers", DL, NT, ["learning_rate", "batch"],
      explain="An optimizer decides how gradients become weight updates. SGD: step = −lr·grad. Momentum: keeps a running velocity to roll through bumps. "
              "Adam: per-parameter adaptive step sizes; a robust default.",
      analogy="SGD walks; momentum is a ball rolling downhill; Adam is a hiker adjusting stride per terrain.",
      questions=[
          Q(1, "Which optimizer keeps a running average of past gradients ('velocity')?", ["Plain SGD", "Momentum", "None", "Dropout"], 1, "Velocity."),
          Q(2, "Adam adapts…", ["The dataset", "The step size per parameter", "The loss function", "Batch size"], 1, "Using gradient moment estimates."),
          Q(3, "Changing the optimizer can change…", ["Only speed", "Speed, stability and the final solution reached", "Nothing", "Only the test set"], 1, "Optimization path matters."),
      ]),

    C("overfitting", "Overfitting", DL, "ml_workshop", ["train_test_split", "decision_trees"],
      explain="Overfitting: the model memorises training noise instead of the general pattern. Signature: training score ≫ validation score. "
              "Remedies: simpler model, regularisation, more data, fewer noisy features, early stopping, dropout. Underfitting is the opposite: both scores low.",
      analogy="A student memorising answers to last year's exam rather than understanding the subject.",
      visual="overfit_curve",
      questions=[
          Q(1, "Train accuracy 99%, validation 62%. Diagnosis?", ["Underfitting", "Overfitting", "Perfect", "Leakage for sure"], 1, "Huge generalisation gap."),
          Q(2, "Train 61%, validation 60%. Diagnosis?", ["Overfitting", "Underfitting (model too simple or features weak)", "Leakage", "Ideal"], 1, "Both low."),
          Q(3, "Which is NOT a remedy for overfitting?", ["More data", "Regularisation", "Making the model much bigger with no constraints", "Early stopping"], 2, "That makes it worse."),
      ], predict="tree_depth"),

    C("dropout", "Dropout", DL, NT, ["overfitting", "neural_networks"],
      explain="Dropout randomly switches off a fraction of neurons during each training step, forcing the network not to rely on any single neuron. "
              "At prediction time all neurons are used. It's a regulariser.",
      analogy="Training a team where random players sit out each practice — everyone must learn to contribute.",
      questions=[
          Q(1, "Dropout is active…", ["Only during training", "Only at prediction", "Always", "Never"], 0, "Disabled at inference."),
          Q(2, "Dropout mainly fights…", ["Underfitting", "Overfitting", "Missing values", "Leakage"], 1, "Regularisation."),
          Q(3, "Dropout 0.9 on a small network will likely…", ["Underfit (too much information thrown away)", "Overfit more", "Have no effect", "Speed up convergence"], 0, "Too strong a regulariser."),
      ], predict="dropout_gap"),

    C("neural_networks", "Neural Networks", DL, NT, ["forward_pass", "activation"],
      explain="A neural network stacks layers of neurons. Hidden layers learn intermediate features; depth and width give capacity to model "
              "complex, curved patterns — at the cost of more data, compute and tuning.",
      analogy="An assembly line where each station refines the product a bit more.",
      visual="nn_diagram",
      questions=[
          Q(1, "Why add hidden layers?", ["Look impressive", "Model non-linear patterns", "Remove data", "Avoid training"], 1, "Capacity."),
          Q(2, "For small tabular datasets, compared with gradient boosting, neural networks are…", ["Always better", "Often not better and harder to tune", "Never usable", "Identical"], 1,
            "Trees often win on small tabular data."),
          Q(3, "A network with 1 hidden layer of 2 ReLU neurons on a spiral dataset will likely…", ["Fit it perfectly", "Underfit", "Overfit", "Diverge"], 1, "Too little capacity for spirals."),
      ], predict="capacity_spiral"),

    C("cnn_intro", "CNN Introduction", DL, VL, ["neural_networks", "matrices"],
      explain="Convolutional neural networks slide small filters (e.g., 3×3) across an image; each filter detects a local pattern like an edge. "
              "The same filter is reused everywhere (weight sharing), so CNNs need far fewer parameters than dense layers on raw pixels.",
      analogy="A magnifying glass scanning a page for a particular shape.",
      visual="conv_filter",
      questions=[
          Q(1, "A convolution filter detects…", ["Global averages", "Local patterns such as edges", "Labels", "Loss"], 1, "Local receptive field."),
          Q(2, "Weight sharing means…", ["Every pixel has its own weights", "The same filter weights are applied at every position", "Weights are copied to other models", "No weights"], 1, "Translation equivariance."),
          N(3, "A 3×3 filter on a 1-channel image has how many weights (excluding bias)?", 9, 0.001, "3·3."),
      ]),

    C("sequence_models", "Sequence Models", DL, "language_center", ["neural_networks"],
      explain="Text, audio and time series are sequences where order matters. RNNs/LSTMs read one step at a time, carrying a hidden state (memory). "
              "They struggle with very long-range dependencies and are slow to train in parallel — motivating transformers.",
      analogy="Reading a novel word by word while keeping a summary in your head.",
      questions=[
          Q(1, "Why does word order matter for 'dog bites man' vs 'man bites dog'?", ["It doesn't", "Same words, different meaning", "Grammar is irrelevant", "Tokenization"], 1, "Sequence matters."),
          Q(2, "An RNN processes a sentence…", ["All at once", "Step by step with a hidden state", "Backwards only", "As an image"], 1, "Recurrent."),
          Q(3, "A key limitation transformers address is…", ["Too few parameters", "Sequential processing and long-range dependencies", "No GPU support for transformers", "Lack of loss"], 1, "Attention connects any two positions directly."),
      ]),

    C("transformers_intro", "Transformers Introduction", DL, "language_center", ["sequence_models", "matrices"],
      explain="Transformers process all tokens in parallel and use *attention* to let each token look at every other token and decide which ones matter. "
              "Positional information is added so order isn't lost. They power modern language models.",
      analogy="A meeting where every participant can directly address anyone else, instead of passing notes down a line.",
      visual="attention_heatmap",
      questions=[
          Q(1, "The core mechanism of transformers is…", ["Convolution", "Attention", "k-means", "Decision trees"], 1, "Self-attention."),
          Q(2, "Why do transformers need positional encodings?", ["To store labels", "Attention by itself ignores word order", "To reduce memory", "To tokenize"], 1, "Permutation invariance."),
          Q(3, "Attention compares every token to every token. Cost grows with sequence length n roughly as…", ["n", "n²", "log n", "constant"], 1, "That's why context windows are limited."),
      ]),
]
