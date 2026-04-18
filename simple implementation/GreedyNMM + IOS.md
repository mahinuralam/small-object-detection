# GreedyNMM + IOS: Detailed Notes

## 1. Purpose in This Pipeline

In tiled inference (SAHI), the same object is often detected multiple times from overlapping tiles. The `GREEDYNMM + IOS` postprocessing step removes cross-tile duplicates before final fusion.

This module is used in:

1. Stage 1 merge: full-image CD-DPA detections + tile detections -> `D_base`
2. Stage 3 merge: SR-TOD detections from weak tiles -> `D_sr`

It is configured by default as:

- `postprocess_type = GREEDYNMM`
- `postprocess_match_metric = IOS`
- `postprocess_match_threshold = 0.5`

## 2. IOS Definition

For two boxes `b_i` and `b_j`:

$$
IOS(b_i,b_j)=\frac{|b_i \cap b_j|}{\min(|b_i|,|b_j|)}
$$

Where:

1. $|b_i \cap b_j|$ is intersection area
2. $|b_i|, |b_j|$ are box areas

### Why IOS instead of IoU in SAHI?

With overlapping tiles, one box may be partial (small) and one may be full (large) for the same object.

- IoU can be low because union is large.
- IOS can still be high if the smaller box is mostly covered.

Example:

1. Full box area = 1000
2. Partial box area = 300
3. Intersection = 270

Then:

- $IoU = 270 / (1000 + 300 - 270) = 270 / 1030 = 0.262$
- $IOS = 270 / 300 = 0.900$

So IOS correctly identifies this as a duplicate pair.

## 3. What GreedyNMM Does in This Codebase

Although called "Non-Maximum Merging", the current implementation behaves as greedy duplicate grouping and representative keeping.

For each class independently:

1. Sort boxes by confidence score (descending)
2. Pick the highest-scoring unmerged box as anchor
3. Compute overlap between anchor and all remaining unmerged boxes using IOS (or IoU if selected)
4. Mark all matches above threshold as duplicates
5. Keep the anchor, discard matched duplicates
6. Repeat until all boxes are processed

Output is a filtered set of boxes/scores/labels, with one representative per duplicate group.

## 4. Mathematical View of the Procedure

For class `c`, let detections be:

$$
\mathcal{D}_c = \{(b_k, s_k)\}_{k=1}^{n_c}
$$

After sorting by score:

$$
s_1 \ge s_2 \ge \dots \ge s_{n_c}
$$

At step `t`, pick highest-score unassigned anchor `b_a` and form match set:

$$
\mathcal{M}_a = \{b_j \mid IOS(b_a,b_j) > \tau\}
$$

Keep only anchor representative from this group, mark all members assigned, and continue.

Final kept set for class `c`:

$$
\hat{\mathcal{D}}_c = \{(b_a,s_a)\text{ from each matched group}\}
$$

All classes are concatenated to produce merged detections.

## 5. Threshold Behavior (`postprocess_match_threshold`)

Let threshold be $\tau$ (default 0.5):

1. Lower $\tau$ (for example 0.4):
- More aggressive duplicate removal
- Fewer leftover duplicates
- Higher risk of collapsing nearby distinct instances in crowded scenes

2. Higher $\tau$ (for example 0.7):
- More conservative matching
- Safer for dense scenes
- More duplicate leftovers possible

Practical recommendation:

- Start with 0.5 for SAHI boundary duplicates
- Sweep 0.45, 0.5, 0.55 for crowded pedestrian-heavy validation subsets

## 6. Difference from Standard NMS

Standard NMS:

1. Uses overlap metric (usually IoU)
2. Suppresses lower-scoring boxes relative to selected boxes

Current GreedyNMM path here:

1. Uses class-wise greedy grouping
2. Uses IOS by default (better for partial/full cross-tile pairs)
3. Keeps one high-confidence representative per matched group

In practice for SAHI, IOS-based matching generally handles boundary-split duplicates better than pure IoU-NMS.

## 7. Where It Is Used in the 4-Stage System

1. Stage 1 merge (`D_base`): full-image CD-DPA + tile CD-DPA detections
2. Stage 3 merge (`D_sr`): weak-tile SR-TOD detections
3. Stage 4 still applies class-wise final NMS between `D_base` and `D_sr`

So there are two dedup layers:

1. Intra-stage tile dedup: GreedyNMM + IOS
2. Inter-stage fusion dedup: class-wise NMS

## 8. Complexity Consideration

For each class with `n` detections, overlap checking is pairwise in the worst case, roughly $O(n^2)$. In practice:

1. Class partitioning reduces pair count
2. Early assignment of matched groups reduces repeated comparisons

## 9. Failure Modes and Mitigation

### 9.1 Over-merging in dense scenes

Symptom:

- close neighboring objects merged as duplicates

Mitigation:

1. Increase IOS threshold
2. Use category-specific thresholds if class density differs strongly

### 9.2 Under-merging tile duplicates

Symptom:

- multiple near-identical boxes remain around tile boundaries

Mitigation:

1. Lower IOS threshold slightly
2. Increase tile overlap consistency or tune detector score threshold

### 9.3 Metric mismatch

Symptom:

- IoU-based behavior too strict for partial boxes

Mitigation:

1. Keep IOS as default for SAHI
2. Use IoU only for ablation comparison

## 10. Suggested Paper Text (Ready to Use)

Short methods sentence:

"We resolve cross-tile duplicates via class-wise greedy matching using Intersection-over-Smaller (IOS, threshold 0.5), retaining the highest-confidence representative per matched group."

Longer paragraph:

"Given heavy overlap in SAHI crops, the same object often appears as a full box in one tile and a clipped box in another. IoU may under-estimate this duplicate relation due to large union area. Therefore, we apply class-wise greedy duplicate matching with IOS, where each high-score anchor absorbs same-class boxes with IOS above threshold and only the anchor is retained. This operation is applied after Stage-1 tile aggregation and again after Stage-3 weak-tile refinement before final cross-stage NMS."

## 11. Minimal Pseudocode

```python
for cls in unique_classes:
    B, S = boxes_of_class(cls), scores_of_class(cls)
    order = argsort_desc(S)
    merged = [False] * len(B)

    for i in order:
        if merged[i]:
            continue
        anchor = B[i]
        keep(anchor)

        for j in order:
            if merged[j]:
                continue
            if IOS(anchor, B[j]) > tau:
                merged[j] = True

        merged[i] = False  # keep anchor
```

## 12. Key Takeaway

`GREEDYNMM + IOS` is your SAHI-specific duplicate-control mechanism. It is especially effective for partial/full duplicate pairs across tile boundaries and is a strong practical choice before final fusion NMS.
