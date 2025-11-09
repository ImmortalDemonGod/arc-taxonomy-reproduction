# Ambiguous Tasks Analysis

Analysis of 14 tasks classified as 'ambiguous' by taxonomy_classifier_v3.py

**Finding:** All 14 tasks have identifiable patterns that COULD be classified.
The classifier returns 'ambiguous' because its decision tree lacks rules for these patterns.

---

## 1. Task `0b148d64`

**Classifier Output:** `ambiguous`

**Should Be:** `C1`

**Reason:** Crops target quadrant using subgrid()

**Generator Code:**

```python
def generate_0b148d64(diff_lb: float, diff_ub: float) -> dict:
    itv = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    bgc = choice(itv)
    remitv = remove(bgc, itv)
    g = canvas(bgc, (h, w))
    x = randint(3, h - 3)
    y = randint(3, w - 3)
    di = randint(2, h - x - 1)
    dj = randint(2, w - y - 1)
    A = backdrop(frozenset({(0, 0), (x, y)}))
    B = backdrop(frozenset({(x + di, 0), (h - 1, y)}))
    C = backdrop(frozenset({(0, y + dj), (x, w - 1)}))
    D = backdrop(frozenset({(x + di, y + dj), (h - 1, w - 1)}))
    cola = choice(remitv)
    colb = choice(remove(cola, remitv))
    trg = choice((A, B, C, D))
    rem = remove(trg, (A, B, C, D))
    subf = lambda bx: {
        choice(totuple(connect(ulcorner(bx), urcorner(bx)))),
        choice(totuple(connect(ulcorner(bx), llcorner(bx)))),
        choice(totuple(connect(urcorner(bx), lrcorner(bx)))),
        choice(totuple(connect(llcorner(bx), lrcorner(bx)))),
    }
    sampler = lambda bx: set(sample(
        totuple(bx),
        len(bx) - unifint(diff_lb, diff_ub, (0, len(bx) - 1))
    ))
    gi = fill(g, cola, sampler(trg) | subf(trg))
    for r in rem:
        gi = fill(gi, colb, sampler(r) | subf(r))
    go = subgrid(frozenset(trg), gi)
    return {'input': gi, 'output': go}


```

---

## 2. Task `1c786137`

**Classifier Output:** `ambiguous`

**Should Be:** `C1`

**Reason:** Crops box interior using crop()

**Generator Code:**

```python
def generate_1c786137(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    num_cols_card_bounds = (1, 8)
    colopts = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    noise_card_bounds = (0, h * w)
    c = canvas(0, (h, w))
    inds = totuple(asindices(c))
    num_noise = unifint(diff_lb, diff_ub, noise_card_bounds)
    num_cols = unifint(diff_lb, diff_ub, num_cols_card_bounds)
    noiseinds = sample(inds, num_noise)
    colset = sample(colopts, num_cols)
    trgcol = choice(difference(colopts, colset))
    noise = frozenset((choice(colset), ij) for ij in noiseinds)
    gi = paint(c, noise)
    boxhrng = (3, max(3, h//2))
    boxwrng = (3, max(3, w//2))
    boxh = unifint(diff_lb, diff_ub, boxhrng)
    boxw = unifint(diff_lb, diff_ub, boxwrng)
    boxi = choice(interval(0, h - boxh + 1, 1))
    boxj = choice(interval(0, w - boxw + 1, 1))
    loc = (boxi, boxj)
    llc = add(loc, toivec(boxh - 1))
    urc = add(loc, tojvec(boxw - 1))
    lrc = add(loc, (boxh - 1, boxw - 1))
    l1 = connect(loc, llc)
    l2 = connect(loc, urc)
    l3 = connect(urc, lrc)
    l4 = connect(llc, lrc)
    l = l1 | l2 | l3 | l4
    gi = fill(gi, trgcol, l)
    go = crop(gi, increment(loc), (boxh - 2, boxw - 2))
    return {'input': gi, 'output': go}


```

---

## 3. Task `25d8a9c8`

**Classifier Output:** `ambiguous`

**Should Be:** `L1`

**Reason:** Logical rule: single-color row → 5, multi-color → 0

**Generator Code:**

```python
def generate_25d8a9c8(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    gi = []
    go = []
    ncols = unifint(diff_lb, diff_ub, (2, 10))
    ccols = sample(cols, ncols)
    for k in range(h):
        singlecol = choice((True, False))
        col = choice(ccols)
        row = repeat(col, w)
        if singlecol:
            gi.append(row)
            go.append(repeat(5, w))
        else:
            remcols = remove(col, ccols)
            nothercinv = unifint(diff_lb, diff_ub, (1, w - 1))
            notherc = w - 1 - nothercinv
            notherc = min(max(1, notherc), w - 1)
            row = list(row)
            indss = interval(0, w, 1)
            for j in sample(indss, notherc):
                row[j] = choice(remcols)
            gi.append(tuple(row))
            go.append(repeat(0, w))
    gi = tuple(gi)
    go = tuple(go)
    return {'input': gi, 'output': go}


```

---

## 4. Task `5582e5ca`

**Classifier Output:** `ambiguous`

**Should Be:** `A2`

**Reason:** Identifies most common color attribute

**Generator Code:**

```python
def generate_5582e5ca(diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    numc = unifint(diff_lb, diff_ub, (2, min(10, h * w - 1)))
    ccols = sample(colopts, numc)
    mostc = ccols[0]
    remcols = ccols[1:]
    leastnummostcol = (h * w) // numc + 1
    maxnummostcol = h * w - numc + 1
    nummostcold = unifint(diff_lb, diff_ub, (0, maxnummostcol - leastnummostcol))
    nummostcol = min(max(leastnummostcol, maxnummostcol - nummostcold), maxnummostcol)
    kk = len(remcols)
    remcount = h * w - nummostcol - kk
    remcounts = [1 for k in range(kk)]
    for j in range(remcount):
        cands = [idx for idx, c in enumerate(remcounts) if c < nummostcol - 1]
        if len(cands) == 0:
            break
        idx = choice(cands)
        remcounts[idx] += 1
    nummostcol = h * w - sum(remcounts)
    gi = canvas(-1, (h, w))
    inds = asindices(gi)
    mclocs = sample(totuple(inds), nummostcol)
    gi = fill(gi, mostc, mclocs)
    go = canvas(mostc, (h, w))
    inds = inds - set(mclocs)
    for col, count in zip(remcols, remcounts):
        locs = sample(totuple(inds), count)
        inds = inds - set(locs)
        gi = fill(gi, col, locs)
    return {'input': gi, 'output': go}


```

---

## 5. Task `6d0160f0`

**Classifier Output:** `ambiguous`

**Should Be:** `A1`

**Reason:** Selects object with yellow marker attribute

**Generator Code:**

```python
def generate_6d0160f0(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (4,))
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nh, nw = h, w
    bgc, linc = sample(cols, 2)
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = canvas(bgc, (fullh, fullw))
    for iloc in range(h, fullh, h+1):
        gi = fill(gi, linc, hfrontier((iloc, 0)))
    for jloc in range(w, fullw, w+1):
        gi = fill(gi, linc, vfrontier((0, jloc)))
    noccs = unifint(diff_lb, diff_ub, (1, h * w))
    denseinds = asindices(canvas(-1, (h, w)))
    sparseinds = {(a*(h+1), b*(w+1)) for a, b in denseinds}
    locs = sample(totuple(sparseinds), noccs)
    trgtl = choice(locs)
    remlocs = remove(trgtl, locs)
    ntrgt = unifint(diff_lb, diff_ub, (1, (h * w - 1)))
    place = choice(totuple(denseinds))
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(cols, ncols)
    candss = totuple(remove(place, denseinds))
    trgrem = sample(candss, ntrgt)
    trgrem = {(choice(ccols), ij) for ij in trgrem}
    trgtobj = {(4, place)} | trgrem
    go = paint(gi, shift(sfilter(trgtobj, lambda cij: cij[0] != linc), multiply(place, increment((h, w)))))
    gi = paint(gi, shift(trgtobj, trgtl))
    toleaveout = ccols
    for rl in remlocs:
        tlo = choice(totuple(ccols))
        ncells = unifint(diff_lb, diff_ub, (1, h * w - 1))
        inds = sample(totuple(denseinds), ncells)
        obj = {(choice(remove(tlo, ccols) if len(ccols) > 1 else ccols), ij) for ij in inds}
        toleaveout = remove(tlo, toleaveout)
        gi = paint(gi, shift(obj, rl))
    return {'input': gi, 'output': go}


```

---

## 6. Task `995c5fa3`

**Classifier Output:** `ambiguous`

**Should Be:** `A1`

**Reason:** Identifies shape patterns and maps to colors

**Generator Code:**

```python
def generate_995c5fa3(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    o1 = asindices(canvas(-1, (4, 4)))
    o2 = box(asindices(canvas(-1, (4, 4))))
    o3 = asindices(canvas(-1, (4, 4))) - {(1, 0), (2, 0), (1, 3), (2, 3)}
    o4 = o1 - shift(asindices(canvas(-1, (2, 2))), (2, 1))
    mpr = [(o1, 2), (o2, 8), (o3, 3), (o4, 4)]
    num = unifint(diff_lb, diff_ub, (1, 6))
    h = 4
    w = 4 * num + num - 1
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    ccols = []
    for k in range(num):
        col = choice(remcols)
        obj, outcol = choice(mpr)
        locj = 5 * k
        gi = fill(gi, col, shift(obj, (0, locj)))
        ccols.append(outcol)
    go = tuple(repeat(c, num) for c in ccols)
    return {'input': gi, 'output': go}


```

---

## 7. Task `a1570a43`

**Classifier Output:** `ambiguous`

**Should Be:** `S2`

**Reason:** Reverses spatial displacement

**Generator Code:**

```python
def generate_a1570a43(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    oh = unifint(diff_lb, diff_ub, (3, h))
    ow = unifint(diff_lb, diff_ub, (3, w))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    crns = {(loci, locj), (loci + oh - 1, locj), (loci, locj + ow - 1), (loci + oh - 1, locj + ow - 1)}
    cands = shift(asindices(canvas(-1, (oh-2, ow-2))), (loci+1, locj+1))
    bgc, dotc = sample(cols, 2)
    remcols = remove(bgc, remove(dotc, cols))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    gipro = canvas(bgc, (h, w))
    gipro = fill(gipro, dotc, crns)
    sp = choice(totuple(cands))
    obj = {sp}
    cands = remove(sp, cands)
    ncells = unifint(diff_lb, diff_ub, (oh + ow - 5, max(oh + ow - 5, ((oh - 2) * (ow - 2)) // 2)))
    for k in range(ncells - 1):
        obj.add(choice(totuple((cands - obj) & mapply(neighbors, obj))))
    while shape(obj) != (oh-2, ow-2):
        obj.add(choice(totuple((cands - obj) & mapply(neighbors, obj))))
    obj = {(choice(ccols), ij) for ij in obj}
    go = paint(gipro, obj)
    nperts = unifint(diff_lb, diff_ub, (1, max(h, w)))
    k = 0
    fullinds = asindices(go)
    while ulcorner(obj) == (loci+1, locj+1) or k < nperts:
        k += 1
        options = sfilter(
            neighbors((0, 0)),
            lambda ij: len(crns & shift(toindices(obj), ij)) == 0 and \
                shift(toindices(obj), ij).issubset(fullinds)
        )
        direc = choice(totuple(options))
        obj = shift(obj, direc)
    gi = paint(gipro, obj)
    return {'input': gi, 'output': go}


```

---

## 8. Task `b94a9452`

**Classifier Output:** `ambiguous`

**Should Be:** `C2`

**Reason:** Swaps colors of nested rectangles using switch()

**Generator Code:**

```python
def generate_b94a9452(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, outer, inner = sample(cols, 3)
    c = canvas(bgc, (h, w))
    oh = unifint(diff_lb, diff_ub, (3, h - 1))
    ow = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    oh2d = unifint(diff_lb, diff_ub, (0, oh // 2))
    ow2d = unifint(diff_lb, diff_ub, (0, ow // 2))
    oh2 = choice((oh2d, oh - oh2d))
    oh2 = min(max(1, oh2), oh - 2)
    ow2 = choice((ow2d, ow - ow2d))
    ow2 = min(max(1, ow2), ow - 2)
    loci2 = randint(loci+1, loci+oh-oh2-1)
    locj2 = randint(locj+1, locj+ow-ow2-1)
    obj1 = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    obj2 = backdrop(frozenset({(loci2, locj2), (loci2 + oh2 - 1, locj2 + ow2 - 1)}))
    gi = fill(c, outer, obj1)
    gi = fill(gi, inner, obj2)
    go = compress(gi)
    go = switch(go, outer, inner)
    return {'input': gi, 'output': go}


```

---

## 9. Task `c3e719e8`

**Classifier Output:** `ambiguous`

**Should Be:** `S3`

**Reason:** Tiles pattern at color positions

**Generator Code:**

```python
def generate_c3e719e8(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(0, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    gob = canvas(-1, (h**2, w**2))
    wg = canvas(-1, (h, w))
    ncols = unifint(diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    nmc = randint(max(1, (h * w) // (ncols + 1) + 1), h * w)
    inds = totuple(asindices(wg))
    mc = choice(cols)
    remcols = remove(mc, cols)
    mcc = sample(inds, nmc)
    inds = difference(inds, mcc)
    gi = fill(wg, mc, mcc)
    ocols = sample(remcols, ncols)
    k = len(inds) // ncols + 1
    for ocol in ocols:
        if len(inds) == 0:
            break
        ub = min(nmc - 1, len(inds))
        ub = min(ub, k)
        ub = max(ub, 1)
        locs = sample(inds, unifint(diff_lb, diff_ub, (1, ub)))
        inds = difference(inds, locs)
        gi = fill(gi, ocol, locs)
    gi = replace(gi, -1, mc)
    o = asobject(gi)
    gob = replace(gob, -1, 0)
    go = paint(gob, mapply(lbind(shift, o), apply(rbind(multiply, (h, w)), ofcolor(gi, mc))))
    return {'input': gi, 'output': go}


```

---

## 10. Task `cce03e0d`

**Classifier Output:** `ambiguous`

**Should Be:** `S3`

**Reason:** Replicates pattern at red cell positions

**Generator Code:**

```python
def generate_cce03e0d(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 8))    
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nred = unifint(diff_lb, diff_ub, (1, h * w - 1))
    ncols = unifint(diff_lb, diff_ub, (1, min(8, nred)))
    ncells = unifint(diff_lb, diff_ub, (1, h * w - nred))
    ccols = sample(cols, ncols)
    gi = canvas(0, (h, w))
    inds = asindices(gi)
    reds = sample(totuple(inds), nred)
    reminds = difference(inds, reds)
    gi = fill(gi, 2, reds)
    rest = sample(totuple(reminds), ncells)
    rest = {(choice(ccols), ij) for ij in rest}
    gi = paint(gi, rest)
    go = canvas(0, (h**2, w**2))
    locs = apply(rbind(multiply, (h, w)), reds)
    res = mapply(lbind(shift, asobject(gi)), locs)
    go = paint(go, res)
    return {'input': gi, 'output': go}


```

---

## 11. Task `cdecee7f`

**Classifier Output:** `ambiguous`

**Should Be:** `A2`

**Reason:** Extracts and arranges color attributes into 3×3 grid

**Generator Code:**

```python
def generate_cdecee7f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    numc = unifint(diff_lb, diff_ub, (1, min(9, w)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    inds = interval(0, w, 1)
    locs = sample(inds, numc)
    locs = order(locs, identity)
    gi = canvas(bgc, (h, w))
    go = []
    for j in locs:
        iloc = randint(0, h - 1)
        col = choice(ccols)
        gi = fill(gi, col, {(iloc, j)})
        go.append(col)
    go = go + [bgc] * (9 - len(go))
    go = tuple(go)
    go = tuple([go[:3], go[3:6][::-1], go[6:]])
    return {'input': gi, 'output': go}


```

---

## 12. Task `d10ecb37`

**Classifier Output:** `ambiguous`

**Should Be:** `C1`

**Reason:** Crops to upper-left 2×2 corner using crop()

**Generator Code:**

```python
def generate_d10ecb37(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = crop(gi, (0, 0), (2, 2))
    return {'input': gi, 'output': go}


```

---

## 13. Task `d4469b4b`

**Classifier Output:** `ambiguous`

**Should Be:** `A1`

**Reason:** Maps background color to pattern

**Generator Code:**

```python
def generate_d4469b4b(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    canv = canvas(5, (3, 3))
    A = fill(canv, 0, {(1, 0), (2, 0), (1, 2), (2, 2)})
    B = fill(canv, 0, corners(asindices(canv)))
    C = fill(canv, 0, {(0, 0), (0, 1), (1, 0), (1, 1)})
    colabc = ((2, A), (1, B), (3, C))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    col, go = choice(colabc)
    gi = canvas(col, (h, w))
    inds = asindices(gi)
    numc = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(cols, numc)
    numcells = unifint(diff_lb, diff_ub, (0, h * w - 1))
    locs = sample(totuple(inds), numcells)
    otherobj = {(choice(ccols), ij) for ij in locs}
    gi = paint(gi, otherobj)
    return {'input': gi, 'output': go}


```

---

## 14. Task `d631b094`

**Classifier Output:** `ambiguous`

**Should Be:** `A2`

**Reason:** Outputs dimension based on cell count

**Generator Code:**

```python
def generate_d631b094(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    bgc = 0
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    nc = unifint(diff_lb, diff_ub, (1, min(30, (h * w) // 2 - 1)))
    c = canvas(bgc, (h, w))
    cands = totuple(asindices(c))
    cels = sample(cands, nc)
    gi = fill(c, fgc, cels)
    go = canvas(fgc, (1, nc))
    return {'input': gi, 'output': go}


```

---

## Summary

| Task ID | Current | Should Be | Reason |
|---------|---------|-----------|--------|
| 0b148d64 | ambiguous | K1 | Crops target quadrant using subgrid() |
| 1c786137 | ambiguous | K1 | Crops box interior using crop() |
| 25d8a9c8 | ambiguous | L1 | Logical rule: single-color row → 5, multi-color → 0 |
| 5582e5ca | ambiguous | A2 | Identifies most common color, outputs canvas(mostc) |
| 6d0160f0 | ambiguous | C2 | Pattern matching: identifies yellow marker, shifts pattern |
| 995c5fa3 | ambiguous | A2 | Identifies shape patterns and maps to output colors |
| a1570a43 | ambiguous | S2 | Reverses spatial displacement to original position |
| b94a9452 | ambiguous | C2 | Swaps colors of nested rectangles using switch() |
| c3e719e8 | ambiguous | S3 | Tiles pattern at positions of dominant color cells |
| cce03e0d | ambiguous | S3 | Replicates pattern at red (color 2) cell positions |
| cdecee7f | ambiguous | A2 | Extracts colors, arranges into 3×3 grid pattern |
| d10ecb37 | ambiguous | K1 | Crops to upper-left 2×2 corner using crop() |
| d4469b4b | ambiguous | A2 | Maps background color attribute to 3×3 pattern |
| d631b094 | ambiguous | A2 | Outputs (1, nc) canvas - dimension based on cell count |

**Classifier Coverage Gap:** 14 tasks (3.5% of 400 tasks)
