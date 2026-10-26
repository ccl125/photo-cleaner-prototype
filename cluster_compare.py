"""离线对比聚类方案：union-find 链式 vs 锚点星型，多阈值。"""
import sqlite3, numpy as np
from collections import Counter

c = sqlite3.connect('/tmp/pc-embed2.sqlite')
rows = c.execute('SELECT id, vec FROM embeddings ORDER BY created_at').fetchall()
V = np.stack([np.frombuffer(b, dtype=np.float32) for _, b in rows])
n = len(V)
S = V @ V.T
iu = np.triu_indices(n, 1)
print(f'n={n}')


def unionfind(th_lo, th_hi=2.0):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    m = (S[iu] > th_lo) & (S[iu] < th_hi)
    for a, b in zip(iu[0][m], iu[1][m]):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    cnt = Counter(find(i) for i in range(n))
    return sorted([s for s in cnt.values() if s >= 2], reverse=True)


def star(th_lo, th_hi=2.0):
    gid = np.full(n, -1)
    m = (S[iu] > th_lo) & (S[iu] < th_hi)
    deg = np.zeros(n)
    np.add.at(deg, iu[0][m], 1)
    np.add.at(deg, iu[1][m], 1)
    g = 0
    for a in np.argsort(-deg):
        if gid[a] >= 0 or deg[a] == 0:
            continue
        row = S[a]
        members = [x for x in range(n) if x != a and gid[x] < 0 and th_lo < row[x] < th_hi]
        if not members:
            continue
        gid[a] = g
        for x in members:
            gid[x] = g
        g += 1
    return sorted(Counter(gid[gid >= 0]).values(), reverse=True), gid


for name, fn, lo, hi in [
    ('union 0.93+', unionfind, 0.93, 2),
    ('union 0.97+', unionfind, 0.97, 2),
    ('star  0.93+', None, 0.93, 2),
    ('star  0.90+', None, 0.90, 2),
]:
    if fn is unionfind:
        sizes = fn(lo, hi)
    else:
        sizes, _ = star(lo, hi)
    print('%s: 组数=%d, 最大组=%d, 涉及照片=%d' %
          (name, len(sizes), sizes[0] if sizes else 0, sum(sizes)))

# star 0.93 的组内凝聚度抽查：每组成员与锚点的最小相似度
sizes, gid = star(0.93, 2.0)
print('\nstar 0.93 组内最小锚点相似度抽查(前5大组):')
for g in [k for k, _ in Counter(gid[gid >= 0]).most_common(5)]:
    members = np.where(gid == g)[0]
    anchor = members[0]  # star 里锚点是度数最大者，这里近似用第一个
    sims = S[anchor, members]
    print('  组 %d: 大小=%d, 成员↔锚 min=%.3f mean=%.3f' % (g, len(members), sims.min(), sims.mean()))
