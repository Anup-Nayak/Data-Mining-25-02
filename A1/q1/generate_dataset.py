import sys, random
import numpy as np

U = int(sys.argv[1])
N = int(sys.argv[2])

# Fixing U = 25 and N = 15000 as per piazza clarifications.
U = 25
N = 15000

random.seed(2026)
np.random.seed(2026)

items = list(range(1, U + 1))

T = max(15, min(50, U // 2))
tpl_size = max(25, min(25, U // 4))
overlap_frac = 0.40

templates = []
for i in range(T):
    if i == 0:
        tpl = set(random.sample(items, min(tpl_size, len(items))))
    else:
        keep_from_prev = int(tpl_size * overlap_frac)
        prev = list(templates[random.randrange(len(templates))])
        take_prev = set(random.sample(prev, min(keep_from_prev, len(prev))))
        need = tpl_size - len(take_prev)
        pool = [x for x in items if x not in take_prev]
        take_new = set(random.sample(pool, min(need, len(pool))))
        tpl = take_prev | take_new
    templates.append(sorted(tpl))

select_template_prob = 0.95
inclusion_prob = 0.95
multi_tpl_prob = 0.22
noise_rate = 0.02

out = []
seen = set()

for _ in range(N):
    txn = set()
    if random.random() < multi_tpl_prob:
        k = random.randint(2, min(4, len(templates)))
        chosen = random.sample(templates, k)
    else:
        chosen = [templates[random.randrange(len(templates))]]
    for tpl in chosen:
        for it in tpl:
            if random.random() < inclusion_prob:
                txn.add(it)
    if random.random() < 0.15:
        extra_tpl = templates[random.randrange(len(templates))]
        for it in extra_tpl:
            if random.random() < 0.6:
                txn.add(it)
    num_noise = max(0, int(noise_rate * U))
    if num_noise > 0:
        noise = random.sample(items, min(num_noise, len(items)))
        for n in noise:
            if random.random() < 0.12:
                txn.add(n)
    desired_min = max(8, min(18, int(np.round(np.random.normal(13, 2)))))
    while len(txn) < desired_min:
        pick = random.choice(items)
        txn.add(pick)
    if len(txn) > U:
        txn = set(random.sample(list(txn), U))
    txn_tuple = tuple(sorted(txn))
    if txn_tuple in seen:
        attempts = 0
        while txn_tuple in seen and attempts < 8:
            if len(txn) < U:
                txn.add(random.choice(items))
            else:
                txn = set(random.sample(items, min(len(items), len(txn))))
            txn_tuple = tuple(sorted(txn))
            attempts += 1
    seen.add(txn_tuple)
    out.append(txn_tuple)

with open("generated_transactions.dat", "w") as f:
    for t in out:
        f.write(" ".join(map(str, t)) + "\n")
