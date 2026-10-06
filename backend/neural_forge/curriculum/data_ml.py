from .schema import C, Q, N

DS, ML, EV = "Data Science", "Machine Learning", "Evaluation"
DD, MW, EC = "data_district", "ml_workshop", "evaluation_chamber"

CONCEPTS = [
    # ---------------- Data Science ----------------
    C("datasets", "Datasets", DS, DD, ["what_is_ai"],
      explain="A dataset is a collection of examples the model learns from. Its quality caps the quality of any model: "
              "garbage in, garbage out. Always ask: *what does this dataset actually contain, and how was it collected?*",
      analogy="A dataset is the textbook the model studies from. A textbook full of errors produces a confused student.",
      visual="table_peek",
      questions=[
          Q(1, "Before training a model, what should you do first?", ["Pick the biggest model", "Inspect the data",
                                                                     "Tune the learning rate", "Deploy it"], 1,
            "Inspection reveals missing values, imbalance, leaks and errors that no algorithm can fix by itself."),
          Q(2, "A face-recognition dataset contains 95% photos of one age group. Likely consequence?",
            ["Equal performance for everyone", "Worse performance on under-represented groups", "Faster training", "No effect"], 1,
            "Models learn what they see; under-represented groups get fewer examples → worse accuracy (a bias issue)."),
          Q(3, "Which question matters most when judging a dataset?", ["What font is it in?",
            "Does it represent the situation where the model will be used?", "Is the file small?", "Is it in CSV?"], 1,
            "A mismatch between training data and deployment conditions (distribution shift) silently breaks models."),
      ]),

    C("rows_columns", "Rows and Columns", DS, DD, ["datasets"],
      explain="In tabular data each row is one example (a student, a house, an email) and each column is one property of it "
              "(hours studied, size, sender). Mixing these up is the root of many beginner bugs.",
      analogy="Rows are people in a class photo; columns are facts on their ID cards.",
      visual="table_peek",
      questions=[
          Q(1, "In a student dataset, one row usually represents…", ["One student", "One property for all students", "The whole class", "The target"], 0,
            "Rows = examples (samples, observations)."),
          Q(2, "A house dataset has 1,000 rows and 12 columns. How many houses?", ["12", "1,000", "12,000", "Unknown"], 1, "One house per row."),
          Q(3, "Your dataset accidentally has the same customer appearing in 40 rows. Main risk?",
            ["None", "The model over-weights that customer and test scores can be inflated if copies land in both train and test",
             "The CSV is invalid", "Columns get deleted"], 1, "Duplicates distort statistics and can leak between train and test splits."),
      ]),

    C("features", "Features", DS, DD, ["rows_columns"],
      explain="Features are the input columns the model uses to make a prediction (study hours, attendance…). "
              "Good features carry information about the outcome and are available *at prediction time*.",
      analogy="Features are the clues a detective uses to solve the case.",
      questions=[
          Q(1, "Predicting exam pass/fail. Which is a reasonable feature?", ["Study hours", "Final exam result", "Student ID number", "Whether they passed"], 0,
            "Final exam result and 'passed' ARE the outcome; an ID number carries no real information."),
          Q(2, "Why is 'student_id' usually a bad feature?", ["It is a number", "It identifies but does not cause or explain the outcome; models may memorise it",
                                                             "IDs are private", "It is too small"], 1, "Arbitrary identifiers let models memorise noise rather than learn patterns."),
          Q(3, "A feature is only known one week AFTER the outcome happens. Should you use it to predict that outcome?",
            ["Yes, more features is better", "No — it won't be available at prediction time (leakage)", "Only with deep learning", "Only if it is numeric"], 1,
            "Features must exist at the moment you make the prediction."),
      ]),

    C("labels", "Labels / Targets", DS, DD, ["features"],
      explain="The label (target) is the answer you want the model to predict: passed (yes/no), price (€), species. "
              "Supervised learning needs labelled examples. Label quality matters as much as features.",
      analogy="Labels are the answer key at the back of the textbook.",
      questions=[
          Q(1, "In 'predict house price from size and location', the label is…", ["Size", "Location", "Price", "The model"], 2, "Price is what we predict."),
          Q(2, "Customer churn: label is 'churned' (0/1). This makes the task…", ["Regression", "Classification", "Clustering", "Not ML"], 1,
            "A categorical target → classification."),
          Q(3, "10% of your spam labels were assigned by a tired annotator at random. Effect?",
            ["Nothing", "Label noise caps achievable accuracy and confuses training", "Model becomes faster", "Improves generalisation"], 1,
            "Noisy labels are like an answer key with mistakes."),
      ]),

    C("missing_values", "Missing Values", DS, DD, ["rows_columns"],
      explain="Real data has holes (NaN, empty cells, 'N/A'). Options: drop rows (loses data), impute with mean/median/most frequent, "
              "or add a 'was_missing' indicator. Many models crash on NaN, so you must decide explicitly.",
      analogy="A survey where some people skipped questions: throw away their whole survey, or make a careful guess?",
      visual="missing_map",
      questions=[
          Q(1, "Most scikit-learn models given NaN values will…", ["Ignore them", "Raise an error", "Treat them as 0", "Delete the column"], 1,
            "You need an explicit strategy such as SimpleImputer (some models like HistGradientBoosting handle NaN natively)."),
          Q(2, "Income has outliers and 5% missing. Better imputation?", ["Mean", "Median", "Max", "Random"], 1, "The median is robust to outliers."),
          Q(3, "Where should the imputation value (e.g., the median) be computed?", ["On the full dataset", "On the training split only",
                                                                                    "On the test split", "It doesn't matter"], 1,
            "Computing statistics on test data leaks information. Fit preprocessing on train, then apply to test."),
      ], predict="missing_strategy"),

    C("categorical", "Categorical Variables", DS, DD, ["features"],
      explain="Categorical columns hold labels, not quantities: city, contract type, colour. Some are ordered (small < medium < large) "
              "— ordinal — and some are not (red, blue) — nominal. Watch for inconsistent spellings ('Male', 'male', 'M').",
      analogy="Jersey numbers are digits but not quantities: player 10 isn't 'twice' player 5.",
      questions=[
          Q(1, "Which column is categorical?", ["Age", "Monthly charges", "Contract type (monthly/yearly)", "Temperature"], 2, "Contract type is a label."),
          Q(2, "Postal codes are stored as numbers. Should a model treat 90210 > 10001 as meaningful?", ["Yes", "No — they are categories", "Only for regression", "Only if sorted"], 1,
            "Numeric-looking codes are still categories."),
          Q(3, "Column 'gender' contains 'F', 'female', 'Female ', 'f'. What problem is this?", ["Missing values", "Inconsistent categories that must be normalised",
                                                                                                 "Outliers", "Leakage"], 1,
            "Without cleaning, a model sees four different categories for one group."),
      ]),

    C("encoding", "Encoding", DS, DD, ["categorical"],
      explain="Models need numbers. One-hot encoding turns 'city' into columns city_Paris, city_Rome (0/1). Ordinal encoding maps "
              "ordered categories to 0, 1, 2. Using ordinal codes for unordered categories invents a fake order.",
      analogy="One-hot is a row of light switches — exactly one is on.",
      visual="onehot",
      questions=[
          Q(1, "One-hot encoding 'colour' ∈ {red, green, blue} creates how many columns?", ["1", "2 or 3", "9", "0"], 1,
            "3 columns (or 2 if you drop one to avoid redundancy)."),
          Q(2, "Encoding {red:0, green:1, blue:2} for a linear model implies…", ["Nothing", "blue is 'twice' green — a fake order", "Better accuracy", "Leakage"], 1,
            "Linear models treat numbers as quantities."),
          Q(3, "A category appears in test data but never in training. A robust OneHotEncoder should…", ["Crash", "Ignore it (all zeros) via handle_unknown='ignore'",
                                                                                                        "Create a new model", "Delete the row"], 1,
            "Production data always surprises you."),
      ]),

    C("scaling", "Feature Scaling", DS, DD, ["features", "descriptive_stats"],
      explain="Features on different scales (income in €10,000s, age in tens) can dominate distance-based models (k-NN, SVM) and slow "
              "gradient descent. Standardisation: (x − mean)/std. Min-max: map to [0, 1]. Trees don't care about scaling.",
      analogy="Comparing athletes by height in millimetres and weight in tonnes: the millimetres would dominate any 'distance'.",
      visual="scaling_demo",
      questions=[
          Q(1, "Which model is MOST affected by feature scale?", ["Decision tree", "k-Nearest Neighbours", "Random forest", "Majority baseline"], 1,
            "k-NN uses distances, which are dominated by large-scale features."),
          N(2, "Standardise x=70 with mean 50 and std 10. Result?", 2, 0.001, "(70 − 50)/10 = 2."),
          Q(3, "Fit the scaler on…", ["All data", "Training data only, then transform test", "Test data", "Each row separately"], 1,
            "Same reason as imputation: avoid leaking test statistics."),
      ], predict="scaling_knn"),

    C("train_test_split", "Train/Test Split", DS, DD, ["labels"],
      explain="We hold out part of the data (e.g., 20%) as a test set the model never sees during training. Test performance estimates "
              "how the model will do on new data. Evaluating on training data is like grading students on questions they memorised.",
      analogy="Practice exam (train) vs real exam (test). Using the same questions on both proves nothing.",
      visual="split_viz",
      questions=[
          Q(1, "Why keep a test set?", ["To train faster", "To estimate performance on unseen data", "To store backups", "Required by Python"], 1,
            "Generalisation is the goal."),
          Q(2, "A model gets 100% on training data. What do we know?", ["It's perfect", "Nothing certain about new data until we check the test set",
                                                                         "It will get 100% on test", "It is underfitting"], 1, "Memorisation also gives 100% train accuracy."),
          Q(3, "With a highly imbalanced label, which split option helps keep class ratios equal in train and test?", ["shuffle=False", "stratify=y",
                                                                                                                       "test_size=0.99", "random_state=None"], 1,
            "Stratified splitting preserves label proportions."),
      ]),

    C("data_viz", "Data Visualization for ML", DS, DD, ["visualization", "datasets"],
      explain="In ML, plot: the target distribution (balance), each feature's histogram (skew, outliers), feature vs target "
              "(signal), and correlations (redundancy, suspicious leaks).",
      analogy="A pilot's pre-flight checklist — but with charts.",
      visual="histogram",
      questions=[
          Q(1, "A feature has correlation 0.999 with the target. First reaction?", ["Great feature!", "Suspicious — check for leakage", "Delete the target", "Ignore"], 1,
            "Near-perfect correlations in real problems usually mean the feature is derived from the target."),
          Q(2, "Which plot best shows class imbalance?", ["Bar chart of label counts", "Scatter plot", "Line chart", "3D surface"], 0, "Count per class."),
          Q(3, "A histogram shows ages of 230 years. What is it likely?", ["A real person", "A data error/outlier to investigate", "Normal", "A label"], 1,
            "Always sanity-check ranges."),
      ]),

    C("data_leakage", "Data Leakage", DS, DD, ["train_test_split", "features"],
      explain="Leakage happens when information that won't be available at prediction time — or comes from the test set — sneaks into training. "
              "Symptoms: suspiciously perfect scores that collapse in production. Typical leaks: target-derived features, future data, preprocessing fit on all data, duplicates across splits.",
      analogy="A student who saw the answer key scores 100% — and learns nothing.",
      questions=[
          Q(1, "Predicting loan default. Feature: 'number of collection calls after default'. This is…", ["A great feature", "Leakage (future information)",
                                                                                                           "A label", "Missing data"], 1,
            "Collection calls only happen after a default — it encodes the answer."),
          Q(2, "You scaled the entire dataset, then split into train/test. Problem?", ["None", "Test statistics leaked into training (mild leakage)",
                                                                                       "The model can't train", "Scaling is unnecessary"], 1, "Split first, then fit preprocessing on train."),
          Q(3, "Your churn model gets 99.9% test accuracy on the first try. Best next step?", ["Deploy immediately", "Investigate features for leakage",
                                                                                               "Add more features", "Increase epochs"], 1,
            "Too-good-to-be-true results deserve suspicion."),
      ], predict="leak_removal"),

    # ---------------- Machine Learning ----------------
    C("supervised", "Supervised Learning", ML, MW, ["labels"],
      explain="Supervised learning learns a mapping from features to a known label using labelled examples. Two main types: "
              "classification (predict a category) and regression (predict a number).",
      analogy="Learning with a teacher who marks every answer.",
      questions=[
          Q(1, "Which is supervised learning?", ["Grouping customers with no labels", "Predicting spam from labelled emails",
                                                  "Compressing images", "Random search"], 1, "Labels = supervision."),
          Q(2, "What does supervised learning require that unsupervised does not?", ["A GPU", "Labelled targets", "More features", "Python"], 1,
            "The label is the defining ingredient."),
          Q(3, "Predicting tomorrow's temperature in °C from today's weather is…", ["Classification", "Regression", "Clustering", "Not ML"], 1, "Continuous numeric target."),
      ]),

    C("unsupervised", "Unsupervised Learning", ML, MW, ["datasets"],
      explain="Unsupervised learning finds structure without labels: grouping similar customers (clustering), compressing features "
              "(dimensionality reduction), spotting unusual points (anomaly detection).",
      analogy="Sorting a pile of unlabelled photos into groups that 'look alike'.",
      questions=[
          Q(1, "Unsupervised learning uses…", ["Only labelled data", "Data without labels", "Only images", "Human graders"], 1, "No targets."),
          Q(2, "Which task is unsupervised?", ["Customer segmentation", "Spam detection", "House price prediction", "Digit recognition with labels"], 0, "No predefined groups."),
          Q(3, "Why is evaluating clustering harder than classification?", ["It is slower", "There is no ground-truth label to compare with",
                                                                            "It uses more memory", "It can't be visualised"], 1,
            "We rely on internal measures (inertia, silhouette) and domain judgement."),
      ]),

    C("regression", "Regression", ML, MW, ["supervised"],
      explain="Regression predicts a continuous number: price, temperature, exam score. Errors are measured as distances "
              "between predicted and actual values (MAE, RMSE).",
      analogy="Guessing the weight of a cake — you are scored on how close you get.",
      visual="scatter_line",
      questions=[
          Q(1, "Which is a regression target?", ["Spam / not spam", "House price in €", "Animal species", "Pass / fail"], 1, "Continuous number."),
          Q(2, "Accuracy is a poor metric for regression because…", ["It is slow", "Exact matches of continuous values almost never happen", "It is too high", "It needs labels"], 1,
            "Predicting €301,250 for a €301,300 house is excellent but 'wrong' under exact match."),
          Q(3, "Predicting the NUMBER of support calls (0, 1, 2, …) is closest to…", ["Regression (count)", "Clustering", "Binary classification", "Unsupervised"], 0,
            "Counts are numeric quantities."),
      ]),

    C("classification", "Classification", ML, MW, ["supervised"],
      explain="Classification predicts a category: spam/ham, pass/fail, cat/dog/bird. Many classifiers output a probability per class; "
              "a threshold (default 0.5 for binary) turns it into a decision.",
      analogy="A mail sorter dropping letters into labelled bins.",
      visual="decision_boundary",
      questions=[
          Q(1, "Predicting whether a transaction is fraud is…", ["Regression", "Binary classification", "Clustering", "Multi-output regression"], 1, "Two classes."),
          Q(2, "Classifying digit images 0–9 is…", ["Binary", "Multiclass classification", "Regression", "Unsupervised"], 1, "Ten classes."),
          Q(3, "Lowering the decision threshold from 0.5 to 0.2 for 'fraud' will usually…", ["Catch more fraud but flag more innocent transactions",
                                                                                             "Catch less fraud", "Change nothing", "Increase accuracy always"], 0,
            "More positives predicted → recall ↑, precision often ↓."),
      ]),

    C("clustering", "Clustering", ML, MW, ["unsupervised", "vectors"],
      explain="Clustering groups similar points. k-means: pick k centres, assign each point to its nearest centre, move each centre to the "
              "mean of its points, repeat until stable. You must choose k.",
      analogy="Placing k ice-cream trucks in a park so that everyone walks as little as possible.",
      visual="kmeans_steps",
      questions=[
          Q(1, "In k-means, k is…", ["Learned automatically", "The number of clusters you choose", "The number of features", "The learning rate"], 1, "A hyperparameter."),
          Q(2, "k-means assigns each point to…", ["A random cluster", "The nearest centroid", "The largest cluster", "Its label"], 1, "Distance-based assignment."),
          Q(3, "Inertia (within-cluster distance) always drops as k increases. So choosing k by minimum inertia gives…", ["The best k", "k = number of points (useless)",
                                                                                                                             "k = 1", "k = 2"], 1,
            "Use the 'elbow' or silhouette score, plus domain sense."),
      ], predict="kmeans_k"),

    C("linear_regression", "Linear Regression", ML, MW, ["regression", "math_functions"],
      explain="Linear regression fits a straight line (or plane): ŷ = w₁x₁ + … + b, choosing weights that minimise squared error. "
              "Fast, interpretable (each weight is 'effect per unit'), but can't capture curves without extra features.",
      analogy="Stretching a rubber band through a cloud of nails so it's as close to all of them as possible.",
      visual="scatter_line",
      questions=[
          Q(1, "Linear regression minimises…", ["Number of wrong labels", "Sum of squared errors", "Tree depth", "Number of features"], 1, "Ordinary least squares."),
          Q(2, "price = 3000·rooms + 50000. The 3000 means…", ["Base price", "Each extra room adds ~€3000 to the prediction", "Number of houses", "Error"], 1, "Coefficient interpretation."),
          Q(3, "Data follows a U-shaped curve. Plain linear regression on x will…", ["Fit perfectly", "Underfit", "Overfit", "Crash"], 1,
            "A straight line cannot bend; add x² as a feature or use a nonlinear model."),
      ], code="from sklearn.linear_model import LinearRegression\nmodel = LinearRegression().fit(X_train, y_train)"),

    C("logistic_regression", "Logistic Regression", ML, MW, ["classification", "linear_regression"],
      explain="Despite its name, logistic regression is a classifier. It computes a linear score z = w·x + b and squashes it with "
              "the sigmoid into a probability. The decision boundary is a straight line (hyperplane). C controls regularisation (smaller C = simpler).",
      analogy="A linear vote: each feature pushes the score up or down; the sigmoid converts the total into a confidence.",
      visual="decision_boundary",
      questions=[
          Q(1, "Logistic regression is used for…", ["Regression only", "Classification", "Clustering", "Image generation"], 1, "It predicts class probabilities."),
          Q(2, "Its decision boundary in 2D is…", ["A circle", "A straight line", "Any shape", "A spiral"], 1, "Linear in the input features."),
          Q(3, "On the 'two moons' dataset, logistic regression will most likely…", ["Separate it perfectly", "Make systematic errors because the boundary must be curved",
                                                                                    "Overfit badly", "Refuse to train"], 1, "Linear models underfit nonlinear patterns."),
      ], code="from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(C=1.0, max_iter=1000)"),

    C("decision_trees", "Decision Trees", ML, MW, ["classification"],
      explain="A decision tree asks a sequence of yes/no questions ('study_hours > 4.5?') to reach a prediction. Very interpretable. "
              "Without limits (max_depth, min_samples_leaf) it can grow until it memorises the training data → overfitting.",
      analogy="A game of 20 Questions.",
      visual="decision_boundary",
      questions=[
          Q(1, "A tree's prediction is made by…", ["Averaging all rows", "Following yes/no splits to a leaf", "A sigmoid", "Random guessing"], 1, "Root → leaf."),
          Q(2, "An unlimited-depth tree reaches 100% train accuracy but 65% test. This is…", ["Underfitting", "Overfitting", "Leakage", "Good generalisation"], 1, "Memorisation."),
          Q(3, "Which setting reduces a tree's tendency to overfit?", ["max_depth=None", "min_samples_leaf=20", "More noise features", "Smaller test set"], 1,
            "Requiring many samples per leaf prevents tiny, memorised leaves."),
      ], predict="tree_depth"),

    C("random_forests", "Random Forests", ML, MW, ["decision_trees"],
      explain="A random forest trains many trees on random subsets of rows and features, then averages their votes. Individual trees overfit in "
              "different ways; averaging cancels much of that noise. Strong default for tabular data, less interpretable than one tree.",
      analogy="Wisdom of the crowd: many imperfect experts voting beat one opinionated expert.",
      questions=[
          Q(1, "A random forest combines…", ["Many neural nets", "Many decision trees", "k-means clusters", "Linear models only"], 1, "Ensemble of trees."),
          Q(2, "Why does a forest usually generalise better than one deep tree?", ["It uses more memory", "Averaging de-correlated trees reduces variance",
                                                                                   "It never overfits", "It uses gradients"], 1, "Variance reduction."),
          Q(3, "Cost of a forest compared to a single tree?", ["Lower latency", "Higher latency/memory and lower interpretability", "No cost", "Needs scaling"], 1,
            "Trade-off: accuracy vs speed and explainability."),
      ], code="from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)"),

    C("knn", "Nearest Neighbours", ML, MW, ["classification", "vectors"],
      explain="k-Nearest Neighbours predicts by finding the k most similar training points and taking a vote (or average). No training "
              "phase — it stores the data. Small k → jagged boundary (overfit); large k → smooth (underfit). Needs scaled features.",
      analogy="'Tell me who your neighbours are and I'll tell you who you are.'",
      visual="decision_boundary",
      questions=[
          Q(1, "k-NN with k=1 classifies a point using…", ["All points", "The single closest training point", "The class mean", "A tree"], 1, "Nearest neighbour."),
          Q(2, "Why is k-NN slow at prediction time on huge datasets?", ["Training is slow", "It must compare against many stored points", "It uses GPUs", "It isn't"], 1,
            "Work happens at query time. Vector databases solve this with approximate search — same idea as RAG retrieval!"),
          Q(3, "Increasing k from 1 to 50 tends to…", ["Make the boundary smoother", "Make it more jagged", "Do nothing", "Cause leakage"], 0, "More neighbours = more averaging."),
      ], predict="knn_k"),

    C("gradient_boosting", "Gradient Boosting", ML, MW, ["random_forests", "gradients"],
      explain="Gradient boosting builds trees *sequentially*: each new small tree fixes the errors (residuals) of the ensemble so far. "
              "Often the best accuracy on tabular data, but needs tuning (learning_rate, n_estimators, depth) and can overfit.",
      analogy="A team of editors: each one corrects the mistakes the previous editors left.",
      questions=[
          Q(1, "Boosting trees are trained…", ["Independently in parallel", "Sequentially, each correcting previous errors", "Without data", "Once"], 1, "Sequential residual fitting."),
          Q(2, "Random forest vs boosting: which averages independent trees?", ["Boosting", "Random forest", "Both", "Neither"], 1, "Forests = bagging; boosting = sequential."),
          Q(3, "Very high learning_rate with many estimators tends to…", ["Underfit", "Overfit", "Have no effect", "Remove features"], 1, "Each tree over-corrects."),
      ]),

    # ---------------- Evaluation ----------------
    C("accuracy", "Accuracy", EV, EC, ["classification"],
      explain="Accuracy = correct predictions / all predictions. Simple and intuitive — but misleading when classes are imbalanced.",
      analogy="A smoke alarm that never rings is 'accurate' 99.9% of the time…",
      questions=[
          N(1, "90 correct out of 120 predictions. Accuracy? (decimal)", 0.75, 0.001, "90/120 = 0.75."),
          Q(2, "98% of transactions are legit. A model always says 'legit'. Accuracy?", ["2%", "50%", "98%", "100%"], 2, "And it catches zero fraud."),
          Q(3, "When is accuracy a reasonable headline metric?", ["Severe imbalance", "Balanced classes with similar error costs",
                                                                  "Regression", "Clustering"], 1, "Accuracy treats all errors equally."),
      ], generators=["acc_from_cm"], predict="imbalance_baseline"),

    C("confusion_matrix", "Confusion Matrix", EV, EC, ["accuracy"],
      explain="A table of actual vs predicted classes. For binary: True Positives (caught), False Positives (false alarms), "
              "False Negatives (missed), True Negatives. Every classification metric is computed from it.",
      analogy="A referee's scorecard that separates 'correct calls' from 'false alarms' and 'missed fouls'.",
      visual="cm_threshold",
      questions=[
          Q(1, "A False Negative in fraud detection is…", ["A legit transaction flagged", "A fraud that was missed", "A correct alarm", "A correct pass"], 1,
            "Predicted negative, actually positive."),
          Q(2, "TP=40, FP=10, FN=20, TN=130. Total samples?", ["200", "170", "60", "50"], 0, "Sum of all cells."),
          Q(3, "For cancer screening, which cell do we most want to minimise?", ["TP", "TN", "FN", "None"], 2, "Missed cancers are the costliest error."),
      ], generators=["acc_from_cm"]),

    C("precision", "Precision", EV, EC, ["confusion_matrix"],
      explain="Precision = TP / (TP + FP): when the model says 'positive', how often is it right? Important when false alarms are costly "
              "(blocking a legitimate email, accusing an innocent customer).",
      analogy="Of all the people the detective arrested, how many were guilty?",
      questions=[
          N(1, "TP=30, FP=10. Precision?", 0.75, 0.001, "30/(30+10)."),
          Q(2, "A spam filter must almost never delete real emails. Prioritise…", ["Recall", "Precision", "Accuracy", "Inertia"], 1, "False positives are the expensive error."),
          Q(3, "Raising the decision threshold usually makes precision…", ["Go up (fewer, more confident positives)", "Go down", "Stay equal", "Undefined"], 0,
            "But recall typically falls."),
      ], generators=["prec_rec"]),

    C("recall", "Recall", EV, EC, ["confusion_matrix"],
      explain="Recall = TP / (TP + FN): of all actual positives, how many did we catch? Important when misses are costly (fraud, disease).",
      analogy="Of all the guilty people in town, how many did the detective arrest?",
      questions=[
          N(1, "TP=30, FN=70. Recall?", 0.3, 0.001, "30/(30+70)."),
          Q(2, "Always predicting 'positive' gives recall of…", ["0", "1.0", "0.5", "Accuracy"], 1, "You catch every positive (with terrible precision)."),
          Q(3, "Precision and recall usually trade off because…", ["They are the same", "Moving the threshold converts FNs into FPs and vice versa",
                                                                   "Of the learning rate", "Of scaling"], 1, "The threshold slider in the confusion-matrix lab shows it."),
      ], generators=["prec_rec"]),

    C("f1", "F1 Score", EV, EC, ["precision", "recall"],
      explain="F1 is the harmonic mean of precision and recall: 2PR/(P+R). It is high only when BOTH are high, so it's a good single "
              "number for imbalanced problems where accuracy lies.",
      analogy="A chain is only as strong as its weakest link — F1 punishes the weaker of P and R.",
      questions=[
          N(1, "Precision 0.5, recall 0.5. F1?", 0.5, 0.001, "2·0.25/1.0 = 0.5."),
          N(2, "Precision 1.0, recall 0.1. F1? (2 decimals)", 0.18, 0.01, "2·0.1/1.1 ≈ 0.18 — far below the arithmetic mean 0.55."),
          Q(3, "Model A: P=0.9, R=0.2. Model B: P=0.6, R=0.6. Higher F1?", ["A", "B", "Equal", "Can't tell"], 1, "A: 0.33, B: 0.60."),
      ], generators=["f1_calc"]),

    C("mae", "MAE", EV, EC, ["regression"],
      explain="Mean Absolute Error = average of |actual − predicted|. Same units as the target ('off by €12k on average'). Robust to outliers.",
      analogy="Average distance by which darts miss the bullseye.",
      questions=[
          N(1, "Errors: +2, −4, +3, −1. MAE?", 2.5, 0.001, "(2+4+3+1)/4 = 2.5."),
          Q(2, "MAE of a house-price model is 15,000. Meaning?", ["15,000 houses wrong", "Predictions are off by about €15,000 on average", "R² = 15,000", "Accuracy 15%"], 1, "Interpretable units."),
          Q(3, "Compared with MSE, MAE penalises a single huge error…", ["More", "Less", "Equally", "Not at all"], 1, "Squaring amplifies large errors; absolute value does not."),
      ], generators=["reg_metrics"]),

    C("mse", "MSE", EV, EC, ["mae"],
      explain="Mean Squared Error = average of (actual − predicted)². Squaring punishes large errors strongly and makes optimisation smooth; "
              "units are squared (€²), which is hard to interpret.",
      analogy="A teacher who deducts points proportional to the square of how wrong you were.",
      questions=[
          N(1, "Errors: 1, −3. MSE?", 5, 0.001, "(1 + 9)/2 = 5."),
          Q(2, "Why is MSE common as a training loss?", ["It's always smaller", "It's smooth and differentiable, heavily penalising big mistakes", "Units are nice", "It ignores outliers"], 1, "Good for gradients."),
          Q(3, "Errors [1,1,1,1] vs [0,0,0,4]. Which has larger MSE?", ["First", "Second", "Same", "Can't tell"], 1, "1 vs 4 — same MAE (1), very different MSE."),
      ], generators=["reg_metrics"]),

    C("rmse", "RMSE", EV, EC, ["mse"],
      explain="RMSE = √MSE. Back in the target's units, but still emphasises big errors more than MAE does.",
      analogy="MSE translated back into the language of the problem.",
      questions=[
          N(1, "MSE = 16. RMSE?", 4, 0.001, "√16."),
          Q(2, "RMSE is always ___ MAE.", ["≤", "≥", "=", "unrelated to"], 1, "Equal only when all errors have the same magnitude."),
          Q(3, "RMSE much larger than MAE suggests…", ["All errors similar", "A few very large errors", "Perfect model", "Leakage"], 1, "Outlier errors inflate RMSE."),
      ], generators=["reg_metrics"]),

    C("r2", "R²", EV, EC, ["mse", "descriptive_stats"],
      explain="R² = 1 − (model's squared error / squared error of always predicting the mean). 1 = perfect, 0 = no better than the mean, "
              "negative = worse than the mean.",
      analogy="How much of the 'mystery' in the target your model explains, compared with a lazy guess.",
      questions=[
          Q(1, "R² = 0 means…", ["Perfect", "No better than predicting the mean", "Worse than mean", "Undefined"], 1, "Baseline-level."),
          Q(2, "Can R² be negative on a test set?", ["No", "Yes — the model is worse than predicting the mean", "Only in training", "Only for classification"], 1, "Common with badly overfit models."),
          N(3, "SSE_model = 20, SSE_mean = 100. R²?", 0.8, 0.001, "1 − 20/100."),
      ]),

    C("validation", "Validation Sets", EV, EC, ["train_test_split"],
      explain="If you tune hyperparameters by checking the test set repeatedly, you slowly overfit to it. Use a third split — validation — for "
              "tuning, and touch the test set only once at the end.",
      analogy="Mock exams (validation) to adjust your study plan; the real exam (test) once.",
      questions=[
          Q(1, "The validation set is used to…", ["Train weights", "Choose hyperparameters / models", "Report final score only", "Store data"], 1, "Tuning."),
          Q(2, "You tried 200 configurations and picked the best by TEST score. The test score is now…", ["Unbiased", "Optimistically biased", "Pessimistic", "Exact"], 1, "Selection leaks into the estimate."),
          Q(3, "A typical split is…", ["100/0/0", "70/15/15 train/val/test", "10/10/80", "50/50/0"], 1, "Exact ratios vary with data size."),
      ]),

    C("cross_validation", "Cross-Validation", EV, EC, ["validation"],
      explain="k-fold cross-validation splits data into k folds, trains k times, each time validating on a different fold, then averages. "
              "More reliable than one split on small data, and the spread (std) shows how stable the estimate is.",
      analogy="Rotating which chapter is used as the quiz so every chapter gets tested once.",
      visual="cv_folds",
      questions=[
          Q(1, "5-fold CV trains the model how many times?", ["1", "5", "10", "25"], 1, "Once per fold."),
          Q(2, "Model A: 0.84 ± 0.06, Model B: 0.85 ± 0.07 (CV). Conclusion?", ["B is clearly better", "The difference is within noise",
                                                                                   "A is clearly better", "Both overfit"], 1, "Overlapping spreads → not a meaningful difference."),
          Q(3, "Main cost of cross-validation?", ["Leakage", "k× the training compute", "Lower accuracy", "Needs GPUs"], 1, "Reliability costs compute."),
      ]),

    C("class_imbalance", "Class Imbalance", EV, EC, ["precision", "recall"],
      explain="When one class is rare (fraud 2%), models can ignore it and still look accurate. Remedies: better metrics (recall, precision, F1, PR curves), "
              "class_weight='balanced', resampling, adjusting the threshold, stratified splits.",
      analogy="Finding needles in a haystack: 'it's all hay' is 98% correct and 100% useless.",
      visual="imbalance_bar",
      questions=[
          Q(1, "Fraud is 2% of data. Which metric is LEAST informative alone?", ["Recall", "F1", "Accuracy", "Precision"], 2, "The majority class dominates accuracy."),
          Q(2, "class_weight='balanced' does what?", ["Deletes data", "Makes mistakes on the rare class cost more during training", "Changes the test set", "Adds features"], 1, "Re-weights the loss."),
          Q(3, "After balancing, recall rose 0.10→0.80 and precision fell 0.90→0.40. Is this better?", ["Always", "Never", "Depends on the cost of misses vs false alarms",
                                                                                                     "Only if accuracy rose"], 2, "Metric choice is a business/ethics decision."),
      ], predict="imbalance_baseline"),
]
