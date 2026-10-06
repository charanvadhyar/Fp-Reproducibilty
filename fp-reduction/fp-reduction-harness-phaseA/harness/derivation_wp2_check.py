"""Brute-force check of the WP2 lemma: over ALL evaluation trees (all shapes, all leaf
assignments) on n <= 6 leaves, the maximum of sum_v (node sum)^2 is attained by the
descending chain, and for mixed-sign data it is bounded by Phi(|x|)."""
import itertools, random, functools

def all_tree_values(vals):
    vals = tuple(vals)
    @functools.lru_cache(None)
    def rec(items):
        if len(items) == 1: return [(items[0], 0.0)]
        out = []; n = len(items); idx = list(range(n))
        for r in range(1, n // 2 + 1):
            for left in itertools.combinations(idx, r):
                right = tuple(i for i in idx if i not in left)
                if r == n - r and left > right: continue
                L = tuple(items[i] for i in left); R = tuple(items[i] for i in right)
                for (sl, ql) in rec(L):
                    for (sr, qr) in rec(R):
                        out.append((sl + sr, ql + qr + (sl + sr) ** 2))
        return out
    return [q for s, q in rec(vals)]

def phi(a):
    xs = sorted(map(abs, a), reverse=True)
    return sum(sum(xs[:k]) ** 2 for k in range(2, len(xs) + 1))

random.seed(1)
for n in (3, 4, 5, 6):
    for _ in range(20):
        x = [random.uniform(0.1, 2.0) for _ in range(n)]
        assert abs(max(all_tree_values(x)) - phi(x)) < 1e-9 * phi(x)
    for _ in range(20):
        x = [random.uniform(-2, 2) for _ in range(n)]
        assert max(all_tree_values(x)) <= phi(x) + 1e-9
print("lemma verified for n <= 6 over all evaluation trees (positive: equality; mixed sign: bound)")
