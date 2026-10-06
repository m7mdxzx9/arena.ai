"""Bundled educational datasets.

All tabular datasets are *synthetic* and generated deterministically from a fixed seed with
documented generative processes, so they are freely distributable and reproducible. The digit
images come from scikit-learn's bundled `load_digits` (UCI optdigits, BSD-licensed via sklearn).

Each dataset entry has: id, title, task (classification/regression/clustering/text_classification),
default target, description, story, and a builder returning a pandas DataFrame.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, make_moons, make_circles, make_blobs

SEED = 42


def _sig(z):
    return 1 / (1 + np.exp(-z))


def student_success(n=600, seed=SEED):
    r = np.random.default_rng(seed)
    hours = np.clip(r.gamma(2.2, 2.0, n), 0, 20).round(1)
    attendance = np.clip(r.normal(82, 11, n), 30, 100).round(0)
    prev = np.clip(r.normal(65, 13, n), 20, 100).round(0)
    sleep = np.clip(r.normal(7, 1.2, n), 3.5, 10).round(1)
    extra = r.choice(["none", "sports", "music", "volunteering"], n, p=[.4, .25, .2, .15])
    parent = r.choice(["high_school", "bachelor", "master", "phd"], n, p=[.4, .35, .18, .07])
    student_id = r.permutation(np.arange(10000, 10000 + n))
    z = 2.6 * (0.38 * (hours - 4.4) + 0.06 * (attendance - 82) + 0.07 * (prev - 65) - 0.25 * (sleep - 7) ** 2 + 0.25 * (extra == "music")
               + r.normal(0, 0.5, n))
    passed = (r.random(n) < _sig(z)).astype(int)
    df = pd.DataFrame(dict(student_id=student_id, study_hours=hours, attendance=attendance, previous_grade=prev,
                           sleep_hours=sleep, extracurricular=extra, parent_education=parent, passed=passed))
    for col, frac in [("sleep_hours", .05), ("attendance", .03)]:
        idx = r.choice(n, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan
    return df


def house_prices(n=500, seed=SEED):
    r = np.random.default_rng(seed + 1)
    hood = r.choice(["riverside", "downtown", "suburbs", "industrial"], n, p=[.2, .25, .4, .15])
    sqm = np.clip(r.normal(105, 35, n), 30, 260).round(0)
    bedrooms = np.clip((sqm / 32 + r.normal(0, .7, n)).round(), 1, 7).astype(int)
    bathrooms = np.clip((bedrooms / 2 + r.normal(0, .5, n)).round(), 1, 4).astype(int)
    age = np.clip(r.exponential(25, n), 0, 120).round(0)
    dist = np.clip(r.gamma(2, 3, n) * np.where(hood == "downtown", .3, 1), 0.2, 40).round(1)
    garage = (r.random(n) < .45).astype(int)
    hood_prem = pd.Series(hood).map({"riverside": 60000, "downtown": 90000, "suburbs": 20000, "industrial": -30000}).values
    price = (55000 + 2350 * sqm + 9000 * bathrooms - 650 * age - 3200 * dist + 14000 * garage + hood_prem
             + r.normal(0, 22000, n))
    return pd.DataFrame(dict(size_sqm=sqm, bedrooms=bedrooms, bathrooms=bathrooms, age_years=age, distance_km=dist,
                             has_garage=garage, neighborhood=hood, price=price.round(-2)))


_SPAM_T = ["Congratulations you have WON a free {prize} click now", "URGENT your account is suspended verify your password at {link}",
           "Earn ${n} per day working from home no experience", "Limited offer cheap {prize} buy now discount", "You are selected for a cash prize claim {link}",
           "Final notice your {prize} delivery fee unpaid pay at {link}", "Hot singles in your area click {link}", "Win big casino bonus {n} free spins today"]
_SPAM_T += ["Hi {name} please review the attached document at {link}", "Your parcel is waiting confirm address {link}",
            "Hello dear friend I need your help with a transfer of ${n}"]
_HAM_T = ["Free pizza in the lounge after the {topic} talk", "Click the link on the course page for the {topic} slides",
          "Offer from the bookshop {n} percent off {topic} textbooks this week",
          "Hi {name} are we still meeting for lunch tomorrow", "The lecture on {topic} moved to room {n}", "Can you send me the notes from {topic} class",
          "Thanks for the help with the {topic} assignment", "Reminder dentist appointment on {day}", "Mum says dinner is at {n} pm on {day}",
          "The project meeting about {topic} is on {day}", "I pushed the fix for the {topic} bug please review", "Happy birthday {name} hope you have a great day",
          "Your library book about {topic} is due on {day}"]


def _fill(t, r):
    return t.format(prize=r.choice(["iPhone", "gift card", "vacation", "laptop", "prize"]), link=r.choice(["bit.ly/x2", "secure-login.co", "claim-now.biz"]),
                    n=int(r.integers(2, 999)), name=r.choice(["Sam", "Lea", "Omar", "Mia", "Ken"]),
                    topic=r.choice(["statistics", "python", "neural networks", "databases", "linear algebra"]),
                    day=r.choice(["Monday", "Tuesday", "Friday", "Sunday"]))


def spam(n=500, seed=SEED):
    r = np.random.default_rng(seed + 2)
    is_spam = (r.random(n) < .35).astype(int)
    texts = []
    for s in is_spam:
        t = _fill(r.choice(_SPAM_T if s else _HAM_T), r)
        if r.random() < .2:  # realistic overlap: ham with spammy words & vice versa
            t += " " + r.choice(["free", "click", "offer", "meeting", "thanks", "now", "today", "please"])
        texts.append(t)
    flip = r.random(n) < .05  # realistic annotation errors: 5% of labels are wrong
    is_spam = np.where(flip, 1 - is_spam, is_spam)
    caps = np.array([sum(ch.isupper() for ch in t) / max(len(t), 1) for t in texts]).round(3)
    links = np.array([int(any(k in t for k in [".co", ".biz", "bit.ly"])) for t in texts])
    return pd.DataFrame(dict(text=texts, caps_ratio=caps, has_link=links, length=[len(t) for t in texts], is_spam=is_spam))


def customer_churn(n=700, seed=SEED):
    r = np.random.default_rng(seed + 3)
    tenure = np.clip(r.exponential(22, n), 1, 72).round(0)
    contract = r.choice(["monthly", "one_year", "two_year"], n, p=[.55, .25, .2])
    monthly = np.clip(r.normal(65, 25, n), 18, 120).round(2)
    calls = r.poisson(1.5, n)
    internet = r.choice(["dsl", "fiber", "none"], n, p=[.35, .45, .2])
    paperless = (r.random(n) < .6).astype(int)
    z = (-0.05 * tenure + 1.1 * (contract == "monthly") - 0.9 * (contract == "two_year") + 0.018 * (monthly - 65)
         + 0.45 * calls + 0.5 * (internet == "fiber") + r.normal(0, .8, n) - 0.9)
    churned = (r.random(n) < _sig(z)).astype(int)
    return pd.DataFrame(dict(tenure_months=tenure, contract=contract, monthly_charges=monthly, support_calls=calls,
                             internet=internet, paperless_billing=paperless, churned=churned))


def digits(seed=SEED):
    d = load_digits()
    df = pd.DataFrame(d.data.astype(int), columns=[f"px_{i}" for i in range(64)])
    df["digit"] = d.target
    return df.sample(frac=1, random_state=seed).reset_index(drop=True).iloc[:900]


def animals(n=500, seed=SEED):
    r = np.random.default_rng(seed + 4)
    specs = {  # class: (weight logmean, height mean, legs, fur p, eggs p, fly p, aquatic p)
        "mammal": (3.0, 60, 4, .95, .02, .03, .08), "bird": (0.0, 25, 2, 0, 1, .85, .15),
        "reptile": (1.0, 15, 4, 0, .95, 0, .3), "fish": (0.3, 10, 0, 0, .95, .01, 1.0), "insect": (-6.0, 1.5, 6, .1, 1, .7, .05)}
    rows = []
    cls = r.choice(list(specs), n, p=[.3, .22, .16, .18, .14])
    for c in cls:
        w, h, legs, fur, eggs, fly, aq = specs[c]
        rows.append(dict(weight_kg=round(float(np.exp(r.normal(w, 1.2))), 3), height_cm=round(max(.2, r.normal(h, h * .4)), 1),
                         legs=int(legs if r.random() > .05 else r.choice([0, 2, 4, 6])), has_fur=int(r.random() < fur),
                         lays_eggs=int(r.random() < eggs), can_fly=int(r.random() < fly), aquatic=int(r.random() < aq), animal_class=str(c)))
    return pd.DataFrame(rows)


def medical(n=600, seed=SEED):
    r = np.random.default_rng(seed + 5)
    age = np.clip(r.normal(52, 15, n), 18, 90).round(0)
    bmi = np.clip(r.normal(27, 5, n), 15, 50).round(1)
    bp = np.clip(r.normal(125, 17, n) + .3 * (age - 50), 85, 200).round(0)
    glucose = np.clip(r.normal(100, 22, n) + .9 * (bmi - 27), 60, 250).round(0)
    chol = np.clip(r.normal(200, 35, n), 110, 320).round(0)
    smoker = (r.random(n) < .22).astype(int)
    activity = r.choice(["low", "medium", "high"], n, p=[.35, .45, .2])
    z = (.045 * (age - 52) + .08 * (bmi - 27) + .035 * (glucose - 100) + .015 * (bp - 125) + .7 * smoker
         - .6 * (activity == "high") + r.normal(0, .9, n) - 1.2)
    risk = (r.random(n) < _sig(z)).astype(int)
    return pd.DataFrame(dict(age=age, bmi=bmi, blood_pressure=bp, glucose=glucose, cholesterol=chol, smoker=smoker,
                             activity_level=activity, condition=risk))


def customer_segments(n=400, seed=SEED):
    r = np.random.default_rng(seed + 6)
    centers = [(25, 20, 75, 12), (55, 25, 20, 2), (85, 45, 82, 9), (40, 55, 50, 5), (90, 30, 18, 3)]
    g = r.integers(0, 5, n)
    data = np.array([r.normal(centers[k], [7, 6, 9, 2]) for k in g])
    return pd.DataFrame(dict(annual_income_k=data[:, 0].clip(8).round(1), age=data[:, 1].clip(18, 80).round(0),
                             spending_score=data[:, 2].clip(1, 100).round(0), visits_per_month=data[:, 3].clip(0).round(0)))


_POS = ["love", "excellent", "great", "perfect", "amazing", "fast", "happy", "recommend", "sturdy", "beautiful"]
_NEG = ["broke", "terrible", "awful", "slow", "refund", "disappointed", "waste", "cheap", "useless", "late"]
_NEU = ["the", "product", "it", "arrived", "box", "colour", "battery", "screen", "size", "price", "was", "and", "with", "after", "a", "week"]


def sentiment(n=500, seed=SEED):
    r = np.random.default_rng(seed + 7)
    y = (r.random(n) < .5).astype(int)
    texts = []
    for i, lab in enumerate(y):
        k = int(r.integers(5, 12))
        words = list(r.choice(_NEU, k))
        main, other = (_POS, _NEG) if lab else (_NEG, _POS)
        for _ in range(int(r.integers(1, 3))):
            words.insert(int(r.integers(0, len(words))), str(r.choice(main)))
        if r.random() < .25:  # mixed reviews make it non-trivial
            words.insert(int(r.integers(0, len(words))), str(r.choice(other)))
        if r.random() < .15:  # negation flips meaning: a genuine bag-of-words weakness
            words.insert(0, "not")
            y[i] = 1 - lab
        texts.append(" ".join(words))
    return pd.DataFrame(dict(review=texts, positive=y))


def fraud(n=5000, seed=SEED):
    r = np.random.default_rng(seed + 8)
    y = (r.random(n) < .02).astype(int)
    amount = np.where(y, r.lognormal(4.7, 1.1, n), r.lognormal(3.6, 1.0, n)).round(2)
    hour = np.where(y, r.choice(24, n, p=np.r_[np.full(6, .07), np.full(18, .58 / 18)]), r.choice(24, n)).astype(int)
    dist = np.where(y, r.gamma(2, 30, n), r.gamma(1.5, 10, n)).round(1)
    foreign = np.where(y, r.random(n) < .4, r.random(n) < .08).astype(int)
    present = np.where(y, r.random(n) < .45, r.random(n) < .7).astype(int)
    velocity = np.where(y, r.poisson(3.4, n), r.poisson(1.2, n))
    return pd.DataFrame(dict(amount=amount, hour=hour, distance_from_home_km=dist, foreign=foreign, card_present=present,
                             tx_last_hour=velocity, is_fraud=y))


# ---------------- Boss datasets ----------------
def overfit_lab(n=140, seed=SEED):
    """Few rows, two genuinely informative features, many noise features."""
    r = np.random.default_rng(seed + 9)
    x1, x2 = r.normal(0, 1, n), r.normal(0, 1, n)
    y = ((x1 + .8 * x2 + r.normal(0, 1.1, n)) > 0).astype(int)
    df = pd.DataFrame(dict(signal_a=x1.round(3), signal_b=x2.round(3)))
    for i in range(30):
        df[f"noise_{i:02d}"] = r.normal(0, 1, n).round(3)
    df["sensor_id"] = r.integers(0, 10 ** 6, n)
    df["label"] = y
    return df


def loan_leak(n=800, seed=SEED):
    r = np.random.default_rng(seed + 10)
    income = np.clip(r.normal(48, 16, n), 10, 150).round(1)
    debt_ratio = np.clip(r.beta(2, 5, n), 0, 1).round(3)
    credit = np.clip(r.normal(660, 70, n), 350, 850).round(0)
    term = r.choice([12, 24, 36, 60], n)
    late_before = r.poisson(.6, n)
    z = -.03 * (income - 48) + 4 * (debt_ratio - .28) - .012 * (credit - 660) + .45 * late_before + .01 * term + r.normal(0, .9, n) - 1.4
    default = (r.random(n) < _sig(z)).astype(int)
    calls_after = np.where(default, r.poisson(7, n) + 1, 0)  # obvious leak: only exists after default
    days_since_payment = np.where(default, r.normal(140, 30, n), r.normal(20, 8, n)).clip(0).round(0)  # subtle leak (measured at audit)
    return pd.DataFrame(dict(income_k=income, debt_ratio=debt_ratio, credit_score=credit, term_months=term,
                             late_payments_before_loan=late_before, collection_calls_after_due=calls_after,
                             days_since_last_payment_at_audit=days_since_payment, defaulted=default))


def data_chaos(n=520, seed=SEED):
    r = np.random.default_rng(seed + 11)
    age = np.clip(r.normal(40, 12, n), 18, 85).round(0)
    income = np.clip(r.lognormal(10.7, .4, n), 12000, 300000).round(0)
    plan = r.choice(["basic", "premium", "family"], n, p=[.5, .3, .2])
    usage = np.clip(r.normal(30, 12, n), 0, 90).round(1)
    tickets = r.poisson(1.2, n)
    z = 1.6 * (-.06 * (usage - 30) + .7 * (tickets - 1.2) + 1.0 * (plan == "basic") - .6 * (plan == "family") - .00002 * (income - 45000) + r.normal(0, .5, n))
    y = (r.random(n) < _sig(z)).astype(int)
    df = pd.DataFrame(dict(age=age.astype(object), income=income, plan=plan.astype(object), monthly_usage_hours=usage,
                           support_tickets=tickets.astype(float), region=r.choice(["north", "south", "east", "west"], n).astype(object), cancelled=y))
    # --- inject chaos ---
    messy = {"basic": ["Basic", "basic ", "BASIC"], "premium": ["Premium", "premium", "PREM"], "family": ["Family", "family "]}
    for i in r.choice(n, 120, replace=False):
        df.at[i, "plan"] = r.choice(messy[df.at[i, "plan"].strip().lower()])
    for i in r.choice(n, 40, replace=False):
        df.at[i, "age"] = r.choice(["N/A", "unknown", "", "?"])
    for i in r.choice(n, 25, replace=False):
        df.at[i, "age"] = str(int(float(df.at[i, "age"]))) + " yrs" if df.at[i, "age"] not in ("N/A", "unknown", "", "?") else df.at[i, "age"]
    for i in r.choice(n, 30, replace=False):
        df.at[i, "income"] = np.nan
    for i in r.choice(n, 8, replace=False):
        df.at[i, "income"] = 9_999_999  # sentinel/outlier
    for i in r.choice(n, 25, replace=False):
        df.at[i, "monthly_usage_hours"] = np.nan
    for i in r.choice(n, 6, replace=False):
        df.at[i, "monthly_usage_hours"] = -1  # impossible
    df["region"] = df["region"].where(r.random(n) > .05, None)
    dup = df.sample(40, random_state=seed)
    return pd.concat([df, dup]).sample(frac=1, random_state=seed).reset_index(drop=True)


def toy_2d(kind: str, n=300, noise=0.25, seed=SEED):
    if kind == "moons":
        X, y = make_moons(n, noise=noise, random_state=seed)
    elif kind == "circles":
        X, y = make_circles(n, noise=noise * .4, factor=.5, random_state=seed)
    elif kind == "blobs":
        X, y = make_blobs(n, centers=3, cluster_std=1.3, random_state=seed)
    elif kind == "xor":
        r = np.random.default_rng(seed)
        X = r.uniform(-1, 1, (n, 2)); y = ((X[:, 0] * X[:, 1]) > 0).astype(int); X += r.normal(0, noise * .3, X.shape)
    elif kind == "spiral":
        r = np.random.default_rng(seed)
        m = n // 2
        t = np.sqrt(r.uniform(0, 1, m)) * 3 * np.pi
        a = np.c_[t * np.cos(t), t * np.sin(t)] / 3 + r.normal(0, noise * .5, (m, 2))
        b = np.c_[-t * np.cos(t), -t * np.sin(t)] / 3 + r.normal(0, noise * .5, (m, 2))
        X = np.r_[a, b]; y = np.r_[np.zeros(m), np.ones(m)].astype(int)
    elif kind == "linear":
        r = np.random.default_rng(seed)
        X = r.normal(0, 1, (n, 2)); y = ((X[:, 0] - .5 * X[:, 1] + r.normal(0, noise, n)) > 0).astype(int)
    else:
        raise ValueError(kind)
    return pd.DataFrame(dict(x1=X[:, 0].round(4), x2=X[:, 1].round(4), label=y))


REGISTRY = {
    "student_success": dict(title="Student Success", task="classification", target="passed", build=student_success, icon="🎓",
                            desc="Predict whether a student passes the final exam.",
                            story="The Academy wants to offer early tutoring to students at risk of failing.",
                            notes="Synthetic. Pass probability depends on study hours, attendance, previous grade, and (non-linearly) sleep. student_id is pure noise. ~5% sleep and ~3% attendance values are missing."),
    "house_prices": dict(title="House Prices", task="regression", target="price", build=house_prices, icon="🏠",
                         desc="Predict a house's sale price.", story="A housing cooperative needs fair price estimates.",
                         notes="Synthetic. Price is linear in size, bathrooms, age, distance, garage and neighbourhood premium, plus Gaussian noise (σ≈22k)."),
    "spam": dict(title="Spam Inbox", task="text_classification", target="is_spam", build=spam, icon="📧",
                 desc="Classify short messages as spam or not.", story="Students are drowning in phishing messages.",
                 notes="Synthetic template-generated messages; ~20% contain cross-over words and 5% of labels are deliberately wrong (annotation noise)."),
    "customer_churn": dict(title="Customer Churn", task="classification", target="churned", build=customer_churn, icon="📉",
                           desc="Predict which telecom customers will leave.", story="The campus ISP wants to keep its subscribers.",
                           notes="Synthetic. Churn driven by short tenure, monthly contracts, high charges, support calls, fiber."),
    "digits": dict(title="Handwritten Digits (8×8)", task="classification", target="digit", build=digits, icon="🔢",
                   desc="Recognise digits 0–9 from 8×8 grayscale images.", story="The Vision Lab's first camera can only see 64 pixels.",
                   notes="900 images from scikit-learn's bundled optdigits dataset (UCI, distributed with scikit-learn under BSD). Pixel values 0–16."),
    "animals": dict(title="Animal Classifier", task="classification", target="animal_class", build=animals, icon="🦊",
                    desc="Classify animals into mammal/bird/reptile/fish/insect from traits.", story="The campus zoo's records lost their labels.",
                    notes="Synthetic multiclass; ~5% leg counts are randomised to create realistic noise."),
    "medical": dict(title="Clinic Risk (synthetic)", task="classification", target="condition", build=medical, icon="🩺",
                    desc="Predict a synthetic condition from vital signs.", story="A purely fictional clinic. NOT medical advice — all data is synthetic.",
                    notes="Synthetic medical-style data. Risk rises with age, BMI, glucose, blood pressure and smoking; high activity lowers it."),
    "customer_segments": dict(title="Customer Segments", task="clustering", target=None, build=customer_segments, icon="🛍️",
                              desc="Discover natural customer groups (no labels).", story="The campus store wants to tailor offers.",
                              notes="Synthetic mixture of 5 Gaussian groups — the 'true' groups are hidden from the player."),
    "sentiment": dict(title="Product Reviews", task="text_classification", target="positive", build=sentiment, icon="⭐",
                      desc="Classify review sentiment.", story="The store wants to know which products disappoint customers.",
                      notes="Synthetic. 25% mixed reviews and 15% with 'not' (which bag-of-words cannot fully understand)."),
    "fraud": dict(title="Card Fraud (imbalanced)", task="classification", target="is_fraud", build=fraud, icon="💳",
                  desc="Detect fraudulent transactions (~2% fraud).", story="The campus card system is being attacked.",
                  notes="Synthetic, ~2% positives. Fraud: larger amounts, night hours, far from home, foreign, card-not-present, high velocity."),
    "overfit_lab": dict(title="Sensor Array (Boss: Overfitter)", task="classification", target="label", build=overfit_lab, icon="📡",
                        desc="140 readings, 33 features — only 2 carry signal.", story="The Overfitter's lair.", boss=True,
                        notes="Label = sign(signal_a + 0.8·signal_b + noise). 30 noise features and an arbitrary sensor_id."),
    "loan_leak": dict(title="Loan Defaults (Boss: The Leak)", task="classification", target="defaulted", build=loan_leak, icon="🏦",
                      desc="Predict loan default at application time.", story="Suspiciously perfect results...", boss=True,
                      notes="Two columns are measured AFTER the outcome: collection calls and days since last payment at audit."),
    "data_chaos": dict(title="Subscription Records (Boss: Data Chaos)", task="classification", target="cancelled", build=data_chaos, icon="🌪️",
                       desc="Messy CRM export: predict cancellations.", story="Someone exported this from three different systems.", boss=True,
                       notes="Injected: duplicates (40), text in numeric 'age', inconsistent plan labels, missing values, sentinel income 9,999,999, negative usage."),
    "moons": dict(title="Two Moons", task="classification", target="label", build=lambda: toy_2d("moons"), icon="🌙", toy=True,
                  desc="2D, curved boundary.", story="", notes="sklearn make_moons, noise 0.25."),
    "circles": dict(title="Circles", task="classification", target="label", build=lambda: toy_2d("circles"), icon="⭕", toy=True,
                    desc="2D, ring inside ring.", story="", notes="sklearn make_circles."),
    "spiral": dict(title="Spiral", task="classification", target="label", build=lambda: toy_2d("spiral"), icon="🌀", toy=True,
                   desc="2D, two interleaved spirals.", story="", notes="Synthetic spirals."),
    "xor": dict(title="XOR", task="classification", target="label", build=lambda: toy_2d("xor"), icon="✖️", toy=True,
                desc="2D, quadrants.", story="", notes="Label = sign(x1·x2)."),
    "linear2d": dict(title="Linearly Separable", task="classification", target="label", build=lambda: toy_2d("linear"), icon="📏", toy=True,
                     desc="2D, straight boundary.", story="", notes="Label = sign(x1 − 0.5·x2 + noise)."),
    "blobs": dict(title="Three Blobs", task="classification", target="label", build=lambda: toy_2d("blobs"), icon="🫧", toy=True,
                  desc="2D, three clusters.", story="", notes="sklearn make_blobs."),
}


@lru_cache(maxsize=64)
def _cached(ds_id: str) -> pd.DataFrame:
    return REGISTRY[ds_id]["build"]()


def load(ds_id: str) -> pd.DataFrame:
    if ds_id not in REGISTRY:
        raise KeyError(f"Unknown dataset '{ds_id}'")
    return _cached(ds_id).copy()


def catalog() -> list[dict]:
    out = []
    for k, v in REGISTRY.items():
        df = _cached(k)
        out.append(dict(id=k, title=v["title"], task=v["task"], target=v["target"], icon=v["icon"], description=v["desc"],
                        story=v["story"], notes=v["notes"], rows=int(len(df)), columns=list(df.columns),
                        boss=bool(v.get("boss")), toy=bool(v.get("toy"))))
    return out
