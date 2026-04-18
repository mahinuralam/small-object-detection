# DGFE Block Diagram (Detailed)

This file provides a standalone, paper-style architecture diagram for the DGFE module
(Difference Map Guided Feature Enhancement) used inside SR-TOD.

It focuses on DGFE internals and its exact connection with ReconstructionHead outputs.

## DGFE Architecture (Detailed)

```mermaid
flowchart TD
    classDef frozen fill:#f2f2f2,stroke:#666,stroke-dasharray: 5 5,color:#111;
    classDef custom fill:#e8f7e8,stroke:#2e7d32,stroke-width:2px,color:#111;
    classDef standard fill:#e8f0ff,stroke:#2f5fb3,color:#111;
    classDef data fill:#fff3e0,stroke:#c77700,color:#111;

    subgraph S0[Inputs from SR-TOD]
        direction TB
        P2[Input feature x (P2)\nN x 256 x Hf x Wf]:::data
        DELTA[Difference map Delta\nN x 1 x Hi x Wi\nDelta = mean(abs(r_img - I), channel)]:::data
        TH[Learnable threshold t\ninit 4/255 ~= 0.0156862]:::custom
    end

    subgraph S1[Stage 1: Filtration (Thresholding)]
        direction TB
        SUB[Delta - t]:::custom
        SIGN[sign(.)]:::custom
        BIN[Binary-like mask\nM = (sign(Delta - t) + 1) * 0.5]:::custom
    end

    subgraph S2[Stage 2: Spatial Guidance]
        direction TB
        RESIZE[Nearest resize to feature size\n(Hf, Wf)]:::custom
        MCH[Channel repeat\nM_c = repeat(M, C=256)]:::custom
    end

    subgraph S3[Stage 3: Channel Reweighting (CBAM-style)]
        direction TB
        GAP[Global Avg Pool(x)]:::custom
        GMP[Global Max Pool(x)]:::custom

        MLP1[Shared MLP\nFlatten -> FC 256->16\nReLU -> FC 16->256]:::custom
        MLP2[Shared MLP\nFlatten -> FC 256->16\nReLU -> FC 16->256]:::custom

        SUM[Elementwise Sum\nA = MLP(avg) + MLP(max)]:::custom
        SIG[Sigmoid\nS = sigma(A)]:::custom
        EXP[Expand S to N x 256 x Hf x Wf]:::custom
    end

    subgraph S4[Fusion and Output]
        direction TB
        XCA[Channel-attended\nX_ca = x * S]:::custom
        BOOST[Difference-guided boost\nY = X_ca * M_c + X_ca]:::custom
        OUT[Enhanced feature y\nN x 256 x Hf x Wf]:::data
    end

    %% Flow
    DELTA --> SUB --> SIGN --> BIN --> RESIZE --> MCH
    TH --> SUB

    P2 --> GAP --> MLP1 --> SUM
    P2 --> GMP --> MLP2 --> SUM
    SUM --> SIG --> EXP

    P2 --> XCA
    EXP --> XCA --> BOOST --> OUT
    MCH --> BOOST
```

## Compact Math View (for caption)

Given:

$$
x \in \mathbb{R}^{N \times C \times H_f \times W_f}, \quad C=256
$$

$$
\Delta \in \mathbb{R}^{N \times 1 \times H_i \times W_i}, \quad t \in \mathbb{R}
$$

1. Filtration mask:

$$
M = \frac{\operatorname{sign}(\Delta - t) + 1}{2}
$$

2. Resize and broadcast:

$$
\tilde{M} = \operatorname{Resize}_{\text{nearest}}(M, H_f, W_f), \quad
M_c = \operatorname{repeat}(\tilde{M}, C)
$$

3. Channel attention (avg + max pooled descriptors through shared MLP):

$$
A = \operatorname{MLP}(\operatorname{GAP}(x)) + \operatorname{MLP}(\operatorname{GMP}(x))
$$

$$
S = \sigma(A), \quad S \in \mathbb{R}^{N \times C \times 1 \times 1}
$$

4. Feature enhancement:

$$
X_{ca} = x \odot S
$$

$$
y = X_{ca} \odot M_c + X_{ca}
$$

Equivalent form:

$$
y = X_{ca} \odot (1 + M_c)
$$

## Notes for Paper Figure

- DGFE is applied on P2 features in SR-TOD (`features['0']`).
- It uses a learnable threshold (initialized as `4/255`) to create a hard spatial prior.
- Spatial prior comes from reconstruction failure regions (difference map).
- Output shape is preserved, so detector interfaces (RPN/ROI) remain unchanged.

---

## Conversation Consolidation: DGFE Specific Clarifications

### 1. What exactly is flowing into DGFE?

Two inputs:

1. P2 feature map from backbone+FPN (`x`, shape `N x 256 x Hf x Wf`)
2. Difference map from ReconstructionHead (`Delta`, shape `N x 1 x Hi x Wi`)

Plus one learnable scalar parameter:

1. `learnable_thresh` initialized around `0.0156862` (`4/255`)

### 2. Is DGFE only spatial attention?

No. DGFE combines:

1. Spatial gating from thresholded difference map
2. Channel reweighting from pooled feature descriptors (avg+max + shared MLP)

Both are fused multiplicatively/additively in final output.

### 3. Why nearest interpolation for the difference mask?

The implementation uses nearest interpolation to preserve sharp binary-like mask boundaries
created by thresholding, avoiding smoothing effects from bilinear interpolation.

### 4. Does DGFE change feature resolution or channel count?

No. DGFE is shape-preserving:

$$
x, y \in \mathbb{R}^{N \times 256 \times H_f \times W_f}
$$

### 5. Is DGFE a separate detector head?

No. DGFE is an internal feature enhancement block inside SR-TOD before RPN/ROI heads.

---

## Full Micro-Block Specification (DGFE)

### A. Filtration Block

Operations:

1. Subtract learnable threshold from difference map
2. Apply sign function
3. Remap to binary-like mask range `[0,1]`

Formula:

$$
M = (\operatorname{sign}(\Delta - t) + 1) \times 0.5
$$

### B. Spatial Guidance Block

Operations:

1. Nearest-neighbor resize from image resolution to feature resolution
2. Repeat mask across channels to match feature tensor

Formula:

$$
M_c = \operatorname{repeat}(\operatorname{Resize}_{\text{nearest}}(M), C)
$$

### C. Channel Attention Block

Operations:

1. Global average pooling and global max pooling on input feature
2. Shared two-layer MLP (`256 -> 16 -> 256`, ReLU between layers)
3. Sum branch outputs and apply sigmoid
4. Expand channel gate to spatial dimensions

Formula:

$$
S = \sigma\left(\operatorname{MLP}(\operatorname{GAP}(x)) + \operatorname{MLP}(\operatorname{GMP}(x))\right)
$$

### D. Output Fusion Block

Operations:

1. Apply channel gate: `X_ca = x * S`
2. Apply spatial boost: `y = X_ca * M_c + X_ca`

Interpretation:

1. `M_c = 0` keeps baseline channel-attended features
2. `M_c = 1` doubles feature amplitude in high-difference regions

---

## Implementation Anchors (for verification)

Key source files:

1. [models/enhancements/dgfe_module.py](models/enhancements/dgfe_module.py)
2. [models/enhancements/reconstruction_head.py](models/enhancements/reconstruction_head.py)
3. [models/srtod_model.py](models/srtod_model.py)

Key code-level facts:

1. DGFE default channel setting is `gate_channels=256`.
2. Default reduction ratio is `16` (`256 -> 16 -> 256` in MLP).
3. Pool types are avg and max, both passed through shared MLP then summed.
4. Threshold is a learnable `nn.Parameter`, initialized to `4/255`.
5. In SR-TOD forward, DGFE enhances `features['0']` (P2) before detection heads.

---

## Suggested Figure Caption (Ready to Use)

"DGFE enhances P2 by combining a thresholded reconstruction-difference spatial prior with CBAM-style channel reweighting. The difference map is filtered by a learnable threshold, resized to feature scale, and fused with channel-attended features via additive boosting, producing shape-preserving enhanced features for unchanged Faster R-CNN heads."












X[P2 feature x\nN x 256 x Hf x Wf]:::in
I[img_inputs I\nN x 3 x Hi x Wi in 0..1]:::in

subgraph RH[ReconstructionHead detailed]
    direction TB
    U1[ConvTranspose2d\nk4 s2 p1\n256 -> 128]:::block
    C11[Conv2d 3x3\n128 -> 128]:::block
    R11[ReLU]:::op
    C12[Conv2d 3x3\n128 -> 128]:::block
    R12[ReLU]:::op

    U2[ConvTranspose2d\nk4 s2 p1\n128 -> 64]:::block
    C21[Conv2d 3x3\n64 -> 64]:::block
    R21[ReLU]:::op
    C22[Conv2d 3x3\n64 -> 64]:::block
    R22[ReLU]:::op

    OC[Conv2d 3x3\n64 -> 3]:::block
    OS[Sigmoid]:::op
    RIMG[r_img\nN x 3 x Hi x Wi]:::out

    U1 --> C11 --> R11 --> C12 --> R12 --> U2 --> C21 --> R21 --> C22 --> R22 --> OC --> OS --> RIMG
end

subgraph DM[Difference map detailed]
    direction TB
    SUBI[Subtract\nr_img - I]:::op
    ABS[Abs]:::op
    MEAN[ReduceMean over channel\nDelta = mean(abs(r_img-I), dim=C)]:::op
    DELTA[Delta\nN x 1 x Hi x Wi]:::out

    SUBI --> ABS --> MEAN --> DELTA
end

subgraph DGFE[DGFE detailed]
    direction TB

    T[Learnable threshold t\ninit 4/255]:::block
    SD[Subtract\nDelta - t]:::op
    SGN[Sign]:::op
    ADD1[Add 1]:::op
    MULH[Multiply 0.5]:::op
    M[M mask in 0..1]:::out
    RSZ[Interpolate nearest\n-> Hf x Wf]:::block
    REP[Repeat channels\nM_c: N x 256 x Hf x Wf]:::block

    GAP[GlobalAvgPool2d]:::block
    GMP[GlobalMaxPool2d]:::block

    FLA[Flatten]:::op
    FLM[Flatten]:::op

    FC1A[FC 256 -> 16]:::block
    FC1M[FC 256 -> 16]:::block
    RA[ReLU]:::op
    RM[ReLU]:::op
    FC2A[FC 16 -> 256]:::block
    FC2M[FC 16 -> 256]:::block

    SUM[Elementwise Sum\nA = avg_path + max_path]:::op
    SIG[Sigmoid]:::op
    USQ[Unsqueeze x2 + Expand\nS -> N x 256 x Hf x Wf]:::block

    XMUL[Multiply\nX_ca = x * S]:::op
    DMUL[Multiply\nX_dm = X_ca * M_c]:::op
    ADDX[Add\nY = X_dm + X_ca]:::op
    Y[Enhanced P2 Y\nN x 256 x Hf x Wf]:::out

    T --> SD --> SGN --> ADD1 --> MULH --> M --> RSZ --> REP

    GAP --> FLA --> FC1A --> RA --> FC2A --> SUM
    GMP --> FLM --> FC1M --> RM --> FC2M --> SUM
    SUM --> SIG --> USQ --> XMUL

    REP --> DMUL
    XMUL --> DMUL --> ADDX --> Y
    XMUL --> ADDX
end

DETH[RPN -> ROI heads]:::out

X --> U1
RIMG --> SUBI
I --> SUBI
DELTA --> SD

X --> GAP
X --> GMP
X --> XMUL

Y --> DETH