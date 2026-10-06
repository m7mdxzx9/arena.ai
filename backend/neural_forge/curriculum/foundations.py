from .schema import C, Q, N

B1, B2 = "Foundations", "Mathematics"
A = "foundation_academy"

CONCEPTS = [
    C("what_is_ai", "What is AI?", B1, A, [],
      explain="Artificial Intelligence is the field of building computer systems that perform tasks we "
              "normally associate with intelligence: recognising images, understanding language, making "
              "decisions. Most modern AI is not hand-written rules; it is software that *learns patterns "
              "from examples*.",
      analogy="A rule-based program is a recipe a chef follows exactly. A learning system is an "
              "apprentice who watches thousands of dishes being made and works out the recipe themselves.",
      visual="ai_venn",
      example="Spam filter, two ways: (1) rules — 'if the email contains FREE MONEY, mark spam'. "
              "(2) learning — show the computer 5,000 emails already labelled spam/not-spam and let it "
              "discover which words matter. Approach (2) adapts when spammers change wording.",
      questions=[
          Q(1, "Which of these is the best description of most modern AI systems?",
            ["Programs with millions of hand-written if/else rules", "Software that learns patterns from example data",
             "Robots with human-like consciousness", "Very fast calculators"], 1,
            "Modern AI is dominated by machine learning: the behaviour is learned from data rather than "
            "hand-coded. Consciousness is not required (or claimed) for AI.",
            "Think about the spam-filter example."),
          Q(2, "A thermostat turns on the heater when temperature < 19°C. Is it a learning system?",
            ["Yes, it reacts to its environment", "No, it follows one fixed rule that never changes with data",
             "Yes, because it uses a sensor", "Only if it is connected to the internet"], 1,
            "Reacting to input is not learning. A learning thermostat would adjust its behaviour from "
            "past data (e.g., learn when you usually come home)."),
          Q(3, "Which task is MOST suited to learning from data rather than writing explicit rules?",
            ["Converting Celsius to Fahrenheit", "Sorting a list of numbers",
             "Recognising handwritten digits from many people", "Adding tax to a price"], 2,
            "Handwriting varies enormously; nobody can write rules for every style. The other tasks have "
            "exact, known formulas."),
      ],
      reflect="Name one app you used today that probably contains a learned model. What data might it have learned from?"),

    C("ai_ml_dl", "AI vs ML vs Deep Learning", B1, A, ["what_is_ai"],
      explain="AI is the broad goal. Machine Learning (ML) is the subset of AI where systems learn from "
              "data. Deep Learning (DL) is the subset of ML that uses neural networks with many layers. "
              "Generative AI (like chatbots) is mostly built with deep learning.",
      analogy="Nested boxes: AI ⊃ Machine Learning ⊃ Deep Learning ⊃ (most of) Generative AI.",
      visual="ai_venn",
      questions=[
          Q(1, "Which statement is true?",
            ["Every AI system uses deep learning", "Deep learning is a subset of machine learning",
             "Machine learning and AI mean exactly the same", "Deep learning does not use data"], 1,
            "DL ⊂ ML ⊂ AI. A chess engine with hand-written search is AI but not ML."),
          Q(2, "A decision tree trained on customer data is an example of…",
            ["AI and ML, but not deep learning", "Deep learning", "Neither AI nor ML", "Only rule-based AI"], 0,
            "A decision tree learns from data (ML) but is not a multi-layer neural network (not DL)."),
          Q(3, "Why did deep learning become dominant after ~2012?",
            ["Neural networks were invented in 2012", "Large datasets + GPUs made training big networks practical",
             "Classical ML stopped working", "Laws required it"], 1,
            "Neural networks date to the 1950s; data scale (e.g., ImageNet) and GPU compute made deep "
            "networks practical and dramatically better on perception tasks."),
      ]),

    C("py_variables", "Python Variables", B1, A, [],
      explain="A variable is a *name* attached to a value so you can reuse it. `hours = 7` stores the "
              "number 7 under the name `hours`. Values have types: int (7), float (7.5), str ('seven'), bool (True).",
      analogy="A labelled jar: the label is the variable name, the content is the value. You can empty "
              "the jar and put something else in it (reassignment).",
      visual="var_boxes",
      example="hours = 7\nhours = hours + 1   # now 8\nname = 'Ada'\nis_student = True",
      questions=[
          Q(1, "After `x = 5` then `x = x * 2`, what is x?", ["5", "10", "x*2", "Error"], 1,
            "The right side is computed first using the old value (5*2=10), then stored back into x."),
          Q(2, "What is the type of `3.0`?", ["int", "float", "str", "bool"], 1,
            "Any number written with a decimal point is a float in Python."),
          Q(3, "a = ' 4'; b = 2. What does `a * b` produce?", ["8", "' 4 4'", "' 4 4' — string repetition gives ' 4 4'", "TypeError"], 2,
            "Multiplying a string by an int repeats it: ' 4' * 2 == ' 4 4'. This is a classic bug when "
            "numbers are read from files as text!"),
      ],
      generators=["py_trace_vars"],
      code="hours = 7\nhours = hours + 1\nprint(hours)"),

    C("py_conditions", "Conditions (if/else)", B1, A, ["py_variables"],
      explain="`if` lets a program choose between paths. Comparisons (`>`, `<`, `==`, `!=`) produce True "
              "or False. `elif` tests another condition; `else` catches everything left.",
      analogy="A fork in a road with a signpost: the condition decides which way you walk.",
      example="score = 72\nif score >= 90:\n    grade = 'A'\nelif score >= 70:\n    grade = 'B'\nelse:\n    grade = 'C'",
      questions=[
          Q(1, "score = 72. Using the example code, what is grade?", ["A", "B", "C", "Error"], 1,
            "72 >= 90 is False; 72 >= 70 is True, so the elif branch runs and later branches are skipped."),
          Q(2, "What does `5 == 5.0` evaluate to in Python?", ["True", "False", "Error", "5"], 0,
            "== compares values; the int 5 and float 5.0 are numerically equal."),
          Q(3, "x = 0. Which branch runs: `if x: A` `else: B`?", ["A", "B", "Both", "Neither"], 1,
            "0, empty strings, empty lists and None are 'falsy' — they behave like False in a condition."),
      ], generators=["py_trace_if"]),

    C("py_loops", "Loops", B1, A, ["py_conditions"],
      explain="Loops repeat work. `for item in collection:` visits each element. `while condition:` "
              "repeats until the condition becomes False. Training a model is literally a loop over epochs!",
      analogy="A loop is a conveyor belt: each item passes the worker once.",
      visual="loop_tracer",
      example="total = 0\nfor n in [3, 5, 2]:\n    total = total + n\nprint(total)  # 10",
      questions=[
          Q(1, "How many times does `for i in range(4):` run its body?", ["3", "4", "5", "Forever"], 1,
            "range(4) produces 0, 1, 2, 3 — four values. Python ranges exclude the end."),
          Q(2, "total=0; for n in [1,2,3]: total += n*n. Final total?", ["6", "14", "9", "36"], 1, "1+4+9 = 14."),
          Q(3, "Which loop never stops?", ["for i in range(10): pass", "i=0\nwhile i < 3: i += 1",
                                           "i=0\nwhile i < 3: print(i)", "for c in 'abc': pass"], 2,
            "Inside the third loop, i is never changed, so `i < 3` stays True forever."),
      ], generators=["py_trace_loop"]),

    C("py_functions", "Functions", B1, A, ["py_loops"],
      explain="A function packages reusable logic: inputs (parameters) → work → output (`return`). "
              "Without `return`, a function gives back `None`. ML libraries are huge collections of functions.",
      analogy="A vending machine: insert inputs, press a button, receive an output. You don't need to see inside.",
      example="def mean(values):\n    return sum(values) / len(values)\n\nmean([2, 4, 9])  # 5.0",
      questions=[
          Q(1, "What does `mean([2, 4, 9])` return?", ["15", "5.0", "3", "None"], 1, "(2+4+9)/3 = 15/3 = 5.0"),
          Q(2, "def f(x): x * 2   — what does f(3) return?", ["6", "None", "3", "Error"], 1,
            "There is no `return`, so the computed 6 is thrown away and the function returns None."),
          Q(3, "Why are functions useful in ML code?",
            ["They make code run on GPUs", "They let you reuse and test the same logic (e.g., a metric) in many places",
             "They are required for loops", "They prevent all bugs"], 1,
            "Packaging logic like `accuracy(y_true, y_pred)` once means you can test it once and trust it everywhere."),
      ]),

    C("py_lists", "Lists", B1, A, ["py_variables"],
      explain="A list stores an ordered sequence: `scores = [70, 85, 92]`. Index from 0: `scores[0]` is 70; "
              "`scores[-1]` is the last item. Slices `scores[0:2]` give a sub-list. `len(scores)` counts items.",
      analogy="A numbered row of lockers starting at locker 0.",
      example="scores = [70, 85, 92]\nscores.append(60)\nscores[1]    # 85\nscores[-1]   # 60",
      questions=[
          Q(1, "scores = [70, 85, 92]. What is scores[1]?", ["70", "85", "92", "Error"], 1, "Indexing starts at 0."),
          Q(2, "a = [1,2,3,4,5]. What is a[1:3]?", ["[1,2,3]", "[2,3]", "[2,3,4]", "[1,2]"], 1,
            "Slices include the start index and exclude the end index."),
          Q(3, "a=[1,2]; b=a; b.append(3). What is a?", ["[1,2]", "[1,2,3]", "[3]", "Error"], 1,
            "b = a does not copy — both names point at the same list object. Use a.copy() to duplicate."),
      ], generators=["py_trace_list"]),

    C("py_dicts", "Dictionaries", B1, A, ["py_lists"],
      explain="A dictionary maps keys to values: `student = {'name': 'Ada', 'hours': 7}`. Look up with "
              "`student['hours']`. Dictionaries are how we pass configurations (hyperparameters) around.",
      analogy="A real dictionary: look up a word (key) to find its definition (value).",
      example="config = {'model': 'random_forest', 'n_estimators': 100}\nconfig['n_estimators']  # 100",
      questions=[
          Q(1, "d = {'a': 1, 'b': 2}. What is d['b']?", ["1", "2", "'b'", "Error"], 1, "Keys look up their values."),
          Q(2, "d = {'a': 1}. What happens with d['z']?", ["Returns None", "KeyError", "Returns 0", "Adds 'z'"], 1,
            "Missing keys raise KeyError. Use d.get('z') to get None (or a default) instead."),
          Q(3, "Which is the most natural structure to store a model's hyperparameters?",
            ["A list like [100, 5, 0.1]", "A dict like {'n_estimators': 100, 'max_depth': 5}", "One long string", "A boolean"], 1,
            "Named keys make configurations self-documenting and reproducible."),
      ]),

    C("numpy_basics", "NumPy", B1, A, ["py_lists", "py_functions"],
      explain="NumPy provides arrays: grids of numbers with fast, *vectorised* operations. `np.array([1,2,3]) * 2` "
              "doubles every element at once — no loop. Arrays have a `shape`, e.g. (100, 4) = 100 rows × 4 columns.",
      analogy="A Python list is a bag of items you process one by one; a NumPy array is a spreadsheet column you transform all at once.",
      example="import numpy as np\nx = np.array([1, 2, 3])\nx * 2        # array([2, 4, 6])\nx.mean()     # 2.0\nnp.zeros((2, 3)).shape  # (2, 3)",
      questions=[
          Q(1, "np.array([1,2,3]) + 10 gives…", ["[1,2,3,10]", "array([11,12,13])", "Error", "16"], 1,
            "NumPy broadcasts the scalar to every element."),
          Q(2, "An array of 200 samples with 5 features has shape…", ["(5, 200)", "(200, 5)", "(1000,)", "(200,)"], 1,
            "Convention in ML: rows = samples, columns = features → (n_samples, n_features)."),
          Q(3, "a = np.array([[1,2],[3,4]]). What is a.sum(axis=0)?", ["array([3, 7])", "array([4, 6])", "10", "array([1,4])"], 1,
            "axis=0 collapses rows → column sums: [1+3, 2+4]."),
      ], generators=["np_shape"]),

    C("pandas_basics", "pandas DataFrames", B1, A, ["numpy_basics", "py_dicts"],
      explain="pandas gives you DataFrames: tables with named columns, like a programmable spreadsheet. "
              "`df['age']` selects a column, `df[df['age'] > 30]` filters rows, `df.describe()` summarises.",
      analogy="A DataFrame is an Excel sheet you can command with code — reproducibly.",
      example="import pandas as pd\ndf = pd.read_csv('students.csv')\ndf.shape\ndf['study_hours'].mean()\ndf[df['passed'] == 1].head()",
      questions=[
          Q(1, "How do you select the column 'age' from DataFrame df?", ["df.age()", "df['age']", "df[age]", "select age from df"], 1,
            "Square brackets with the column name as a string (df.age also works for simple names)."),
          Q(2, "What does df[df['score'] > 50] return?", ["The column score", "Rows where score > 50",
                                                          "True/False", "The number of rows"], 1,
            "The inner expression creates a True/False mask; indexing with it keeps matching rows."),
          Q(3, "df.shape is (500, 8). What does that mean?", ["8 rows, 500 columns", "500 rows, 8 columns",
                                                             "500 missing values", "4000 rows"], 1, "shape = (rows, columns)."),
      ]),

    C("visualization", "Visualization", B1, A, ["pandas_basics"],
      explain="Charts reveal patterns that tables hide. Histograms show a distribution of one variable; "
              "scatter plots show the relationship between two; bar charts compare categories; line charts show change over time (e.g., loss per epoch).",
      analogy="A chart is a map; a table is a list of GPS coordinates. Both contain the data — only one is easy to read.",
      visual="histogram",
      questions=[
          Q(1, "Best chart to see whether study hours relate to exam score?", ["Pie chart", "Scatter plot", "Single bar", "Table"], 1,
            "Scatter plots show one variable against another, revealing trends and outliers."),
          Q(2, "Best chart to see how training loss changes across epochs?", ["Line chart", "Pie chart", "Histogram", "Venn diagram"], 0,
            "Loss is a sequence over time — a line chart is the natural choice."),
          Q(3, "Four datasets have identical mean, variance and correlation (Anscombe's quartet). The lesson is…",
            ["Statistics are useless", "Always plot your data — summaries can hide very different shapes",
             "Use only the mean", "Correlation proves causation"], 1,
            "Anscombe's quartet shows identical summary stats for wildly different patterns."),
      ]),

    # ---------------- Mathematics ----------------
    C("algebra", "Basic Algebra", B2, A, [],
      explain="Algebra uses symbols for unknown or changing numbers. In y = 2x + 1, if x = 3 then y = 7. "
              "ML models are formulas whose numbers (parameters) are learned.",
      analogy="A formula is a machine with a dial: turn x and y changes in a predictable way.",
      questions=[
          N(1, "If y = 2x + 1 and x = 4, what is y?", 9, 0.001, "2·4 + 1 = 9."),
          N(2, "Solve for x: 3x − 6 = 9", 5, 0.001, "Add 6: 3x = 15, divide by 3: x = 5."),
          N(3, "A model predicts price = 150·size + 20000. Predicted price for size 80?", 32000, 0.5, "150·80 + 20000 = 12000 + 20000 = 32000."),
      ], generators=["alg_linear"]),

    C("math_functions", "Functions (math)", B2, A, ["algebra"],
      explain="A mathematical function maps each input to exactly one output: f(x) = x². A model is a function from "
              "features to predictions. The slope tells how fast output changes as input changes.",
      analogy="A function is a reliable translator: same input sentence, same translated output.",
      visual="function_plot",
      questions=[
          N(1, "f(x) = x² − 1. What is f(3)?", 8, 0.001, "9 − 1 = 8."),
          Q(2, "Which line rises faster: y = 0.5x or y = 3x?", ["y = 0.5x", "y = 3x", "Same", "Neither rises"], 1,
            "The coefficient of x is the slope: 3 > 0.5."),
          Q(3, "The sigmoid σ(z)=1/(1+e^−z) outputs values in which range?", ["(−∞, ∞)", "(0, 1)", "(−1, 1)", "[0, ∞)"], 1,
            "That is why sigmoid outputs can be read as probabilities."),
      ]),

    C("vectors", "Vectors", B2, A, ["algebra"],
      explain="A vector is an ordered list of numbers, e.g., a student = [7 hours, 90% attendance, 65 grade]. "
              "Each data row is a vector. The dot product a·b = Σ aᵢbᵢ measures alignment; it powers neurons and similarity search.",
      analogy="A vector is an arrow in space — or simply a row of a spreadsheet.",
      visual="vector_plot",
      questions=[
          N(1, "a=[1,2], b=[3,4]. Compute the dot product a·b.", 11, 0.001, "1·3 + 2·4 = 11."),
          N(2, "Length (norm) of [3,4]?", 5, 0.001, "√(3² + 4²) = √25 = 5."),
          Q(3, "Cosine similarity between [1,0] and [0,1] is…", ["1", "0", "−1", "0.5"], 1,
            "They are perpendicular: dot product 0 → cosine 0 → 'unrelated'. Embedding search uses this idea."),
      ], generators=["vec_dot"]),

    C("matrices", "Matrices", B2, A, ["vectors"],
      explain="A matrix is a 2D grid of numbers. A dataset is a matrix (rows=samples, cols=features). "
              "Multiplying a matrix by a vector computes many dot products at once — this is how a neural network layer works.",
      analogy="If a vector is a single row in a spreadsheet, a matrix is the whole sheet.",
      questions=[
          Q(1, "A (3×2) matrix has…", ["3 rows, 2 columns", "2 rows, 3 columns", "6 rows", "5 cells"], 0, "Rows × columns."),
          Q(2, "Can a (3×2) matrix be multiplied by a (2×4) matrix?", ["Yes, result 3×4", "Yes, result 2×2", "No", "Yes, result 4×3"], 0,
            "Inner dimensions must match (2 and 2); the result takes the outer ones (3×4)."),
          N(3, "[[1,2],[3,4]] · [1,1] — what is the SECOND element of the result?", 7, 0.001, "Row 2 · [1,1] = 3 + 4 = 7."),
      ], generators=["mat_shape"]),

    C("probability", "Probability", B2, A, ["algebra"],
      explain="Probability measures how likely something is, from 0 (impossible) to 1 (certain). Classifiers often output "
              "probabilities (\"0.83 chance of spam\"); we then choose a threshold to turn them into decisions.",
      analogy="A weather forecast of 70% rain: not a promise, a calibrated belief.",
      questions=[
          N(1, "A fair die: probability of rolling a 6? (decimal)", 1/6, 0.01, "1 favourable outcome out of 6 → ≈ 0.167."),
          N(2, "P(rain)=0.3. What is P(no rain)?", 0.7, 0.001, "Complement rule: 1 − 0.3."),
          Q(3, "A disease affects 1% of people. A test is 99% accurate. You test positive. Is P(disease) ≈ 99%?",
            ["Yes", "No — it's closer to 50% because healthy people vastly outnumber sick ones", "It's 1%", "It's 0%"], 1,
            "Base rates matter (Bayes' rule): ~0.99% true positives vs ~0.99% false positives → about 50%. Same trap as accuracy on imbalanced data!"),
      ], generators=["prob_basic"]),

    C("descriptive_stats", "Descriptive Statistics", B2, A, ["algebra"],
      explain="Summaries of data: mean (average), median (middle value), standard deviation (typical spread), min/max. "
              "The median resists outliers; the mean does not.",
      analogy="Ten people in a café earn ~€3k; a billionaire walks in. The mean salary explodes; the median barely moves.",
      visual="histogram",
      questions=[
          N(1, "Mean of [2, 4, 6, 8]?", 5, 0.001, "20 / 4 = 5."),
          N(2, "Median of [1, 3, 100]?", 3, 0.001, "Sorted middle value: 3 (mean would be 34.7)."),
          Q(3, "Two classes both have mean score 70; class A has std 2, class B std 20. Which is true?",
            ["Class A scores are more spread out", "Class B scores vary much more", "They are identical", "Std can't be compared"], 1,
            "Standard deviation measures spread around the mean."),
      ], generators=["stats_mean_median"]),

    C("distributions", "Distributions", B2, A, ["descriptive_stats", "probability"],
      explain="A distribution describes how often each value occurs. The normal (bell) curve is symmetric; incomes are "
              "right-skewed (long tail); class labels have a categorical distribution (e.g., 98% / 2%).",
      analogy="Pour sand through a funnel onto a table: the shape of the pile is the distribution.",
      visual="distribution_plot",
      questions=[
          Q(1, "A histogram with a long tail to the right is…", ["Symmetric", "Right-skewed", "Left-skewed", "Uniform"], 1,
            "Skew is named after the side with the long tail."),
          Q(2, "For a normal distribution, roughly what fraction lies within 1 std of the mean?", ["50%", "68%", "95%", "99.7%"], 1,
            "68–95–99.7 rule: 1, 2, 3 standard deviations."),
          Q(3, "For right-skewed data like income, which is usually larger?", ["Median", "Mean", "They are always equal", "Mode"], 1,
            "The long tail of large values pulls the mean above the median."),
      ]),

    C("derivatives", "Derivatives (intuition)", B2, A, ["math_functions"],
      explain="A derivative is the slope of a function at a point: how much the output changes for a tiny change in input. "
              "If the slope of the loss is positive, decreasing the parameter reduces the loss. No proofs needed — just slopes.",
      analogy="Standing on a hill: the derivative tells how steep the ground is under your feet and which way is downhill.",
      visual="gd_descent",
      questions=[
          Q(1, "At the bottom of a U-shaped curve, the slope is…", ["Very positive", "Zero", "Very negative", "Undefined"], 1,
            "The curve is flat at its minimum — which is why optimisation looks for slope ≈ 0."),
          N(2, "f(x) = x². The slope (derivative) is 2x. What is the slope at x = 3?", 6, 0.001, "2·3 = 6."),
          Q(3, "Loss L(w) has slope +4 at the current w. To reduce L you should…", ["Increase w", "Decrease w", "Keep w", "Set w=4"], 1,
            "Positive slope means L grows as w grows, so step the opposite way."),
      ], generators=["deriv_slope"]),

    C("gradients", "Gradient (intuition)", B2, A, ["derivatives", "vectors"],
      explain="With many parameters, the gradient is the vector of all slopes — it points in the direction of steepest increase. "
              "Gradient descent steps in the *opposite* direction: w_new = w − learning_rate × gradient.",
      analogy="Blindfolded on a mountain, you feel the slope in every direction and step downhill. Repeat.",
      visual="gd_descent",
      questions=[
          Q(1, "Gradient descent moves parameters…", ["In the direction of the gradient", "Opposite to the gradient",
                                                       "Randomly", "Only when loss is zero"], 1,
            "The gradient points uphill; we want to go downhill."),
          N(2, "w = 2.0, gradient = 0.5, learning rate = 0.1. New w after one step?", 1.95, 0.001, "2.0 − 0.1·0.5 = 1.95."),
          Q(3, "If the learning rate is far too large, gradient descent may…", ["Converge faster always", "Overshoot and diverge",
                                                                             "Stop immediately", "Become exact"], 1,
            "Huge steps jump across the valley and can bounce higher each time."),
      ], generators=["gd_step"], predict="lr_explosion"),
]
