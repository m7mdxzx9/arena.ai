import numpy as np

from neural_forge import nn


def test_backprop_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 3))
    Y = np.eye(2)[rng.integers(0, 2, 6)]
    for act in ("relu", "tanh", "sigmoid"):
        net = nn.MLP([3, 5, 2], activation=act, seed=1, l2=0.01)
        P, cache = net.forward(X)
        gW, gb = net.backward(cache, Y)
        eps = 1e-6
        for li in range(len(net.W)):
            for idx in [(0, 0), (1, 1)]:
                old = net.W[li][idx]
                net.W[li][idx] = old + eps; lp = net.loss(net.forward(X)[0], Y)
                net.W[li][idx] = old - eps; lm = net.loss(net.forward(X)[0], Y)
                net.W[li][idx] = old
                assert abs((lp - lm) / (2 * eps) - gW[li][idx]) < 1e-5, (act, li, idx)


def test_training_learns_moons():
    r = nn.train(dict(dataset="moons", hidden=[16, 16], optimizer="adam", lr=0.01, epochs=150))
    h = r["history"]
    assert h["train_loss"][-1] < h["train_loss"][0] * 0.5
    assert r["final"]["val_acc"] > 0.9


def test_linear_activation_cannot_solve_circles():
    r = nn.train(dict(dataset="circles", hidden=[32, 32], activation="linear", optimizer="adam", lr=0.01, epochs=150))
    assert r["final"]["val_acc"] < 0.7
    assert any(d["code"] == "linear" for d in r["diagnosis"])


def test_bad_learning_rates_are_diagnosed():
    hot = nn.train(dict(dataset="moons", hidden=[16, 16], optimizer="sgd", lr=20, epochs=60))
    assert {d["code"] for d in hot["diagnosis"]} & {"unstable", "diverged"}
    cold = nn.train(dict(dataset="moons", hidden=[16, 16], optimizer="sgd", lr=0.0005, epochs=60))
    assert {d["code"] for d in cold["diagnosis"]} & {"slow", "underfit"}


def test_pytorch_code_is_generated():
    assert "nn.Linear(2, 8)" in nn.train(dict(dataset="moons", epochs=5))["code"]
