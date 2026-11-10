import math, os, random, statistics, subprocess, sys, time
sys.path.insert(0, "micrograd")            # the cloned repo
from micrograd.nn import MLP

f = lambda x: 3 * x * x + 7 * x + 2
TRAIN = [-1 + 0.25 * i for i in range(9)]
TEST = [-0.97 + 0.1 * i for i in range(20)]   # 20 unseen points inside [-1, 1]
EPOCHS = 200
LRS = [0.001, 0.003, 0.01, 0.02]
SEEDS = range(5)                               # random init for BOTH models
EXE = os.path.abspath("kan_bench") + (".exe" if os.name == "nt" else "")


def run_micrograd(lr, seed):
    random.seed(seed)
    model = MLP(2, [6, 6, 1])                  # 67 params, close to the KAN's 70
    losses = []
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        total = 0.0
        for x in TRAIN:                        # same per-sample SGD as the KAN
            loss = (model([x, x]) - f(x)) ** 2
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data -= lr * p.grad
            total += loss.data
        losses.append(total / len(TRAIN))
    secs = time.perf_counter() - t0
    test = sum((model([x, x]).data - f(x)) ** 2 for x in TEST) / len(TEST)
    return dict(losses=losses, secs=secs, test=test, n=len(model.parameters()))


def run_kan(lr, seed):
    out = subprocess.run([EXE, str(lr), str(EPOCHS), str(seed)], capture_output=True, text=True).stdout.split("\n")
    losses = [float(l.split()[3]) for l in out if l.startswith("step")]
    get = lambda key: float(next(l for l in out if l.startswith(key)).split()[1])
    return dict(losses=losses, secs=get("TIME"), test=get("TEST"), n=int(get("NPARAMS")))


def first_below(losses, thr):
    return next((i + 1 for i, l in enumerate(losses) if l < thr), None)


def med(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return statistics.median(vals) if vals else float("nan")


def summarize(name, runs, lr):
    ok = [r for r in runs if not math.isnan(r["losses"][-1]) and r["losses"][-1] < 1e6]
    ep = [first_below(r["losses"], 0.1) for r in runs]
    return dict(
        name=name, lr=lr, n=runs[0]["n"], diverged=len(runs) - len(ok), runs=len(runs),
        l10=med([r["losses"][9] for r in ok]), l50=med([r["losses"][49] for r in ok]),
        lend=med([r["losses"][-1] for r in ok]), test=med([r["test"] for r in ok]),
        worst=max([r["test"] for r in ok], default=float("nan")), ep=med(ep), secs=med([r["secs"] for r in runs]),
    )


rows = []
for lr in LRS:
    rows.append(summarize("KAN", [run_kan(lr, s) for s in SEEDS], lr))
    rows.append(summarize("micrograd MLP", [run_micrograd(lr, s) for s in SEEDS], lr))

print(f"task: y=3x^2+7x+2, 9 train pts, {EPOCHS} epochs of per-sample SGD, random init, median of {len(SEEDS)} seeds, test = 20 unseen points in [-1,1]\n")
print(f"{'model':<14}{'lr':>7}{'params':>7}{'loss@10':>10}{'loss@50':>10}{'loss@end':>10}{'ep<0.1':>8}{'test mse':>11}{'worst':>9}{'sec':>8}{'div':>5}")
for r in rows:
    print(f"{r['name']:<14}{r['lr']:>7}{r['n']:>7}{r['l10']:>10.4f}{r['l50']:>10.4f}{r['lend']:>10.5f}"
          f"{r['ep']:>8.0f}{r['test']:>11.5f}{r['worst']:>9.4f}{r['secs']:>8.3f}{r['diverged']:>3}/{r['runs']}")
print("\nbest lr per model (lowest final train loss):")
for name in ("KAN", "micrograd MLP"):
    best = min((r for r in rows if r["name"] == name and not math.isnan(r["lend"])), key=lambda r: r["lend"])
    print(f"  {name:<14} lr={best['lr']}  loss={best['lend']:.5f}  test mse={best['test']:.5f}  time={best['secs']:.3f}s")