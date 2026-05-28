# Anti-Money Laundering Detection with Graph Neural Networks and Manual Feature Engineering

**Course:** Graph Mining and Applications  
**Institution:** Sapienza University of Rome  
**Dataset:** IBM AML LI-Small — Altman et al. (2023)  
**Reference:** *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*, arXiv:2306.16424v3

---

## Table of Contents

1. [Introduction and Problem Statement](#1-introduction-and-problem-statement)
2. [GNNs and the 1-WL Expressiveness Limitation](#2-gnns-and-the-1-wl-expressiveness-limitation)
3. [Dataset Description](#3-dataset-description)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Data Preparation and Feature Engineering (Baseline)](#5-data-preparation-and-feature-engineering-baseline)
6. [Model Architecture](#6-model-architecture)
7. [Training Setup](#7-training-setup)
8. [Baseline Results](#8-baseline-results)
9. [GFP Feature Engineering](#9-gfp-feature-engineering)
10. [RWPE Positional Encoding](#10-rwpe-positional-encoding)
11. [GFP Split Projection Architecture](#11-gfp-split-projection-architecture)
12. [Results: Full Comparison](#12-results-full-comparison)
13. [Analysis and Discussion](#13-analysis-and-discussion)
14. [Future Work](#14-future-work)
15. [Conclusions](#15-conclusions)

---

## 1. Introduction and Problem Statement

### 1.1 Anti-Money Laundering in Financial Networks

Money laundering is the process of concealing the origins of illegally obtained funds by passing them through a complex sequence of banking transfers or commercial transactions. The UN estimates that between $0.8 and $2 trillion USD are laundered globally every year — roughly 2–5% of global GDP. Despite this scale, less than 1% of criminal proceeds are detected and seized by authorities, and current rule-based detection systems suffer from false positive rates exceeding 95%, generating enormous operational cost for financial institutions.

The money laundering process traditionally follows three stages:

- **Placement:** Illicit funds enter the financial system (e.g., cash deposits, purchase of assets, use of front businesses).
- **Layering:** Funds are moved through a complex web of transactions — wire transfers, shell companies, currency conversions, multiple jurisdictions — to obscure the audit trail. This is the stage where the characteristic graph patterns emerge.
- **Integration:** Laundered money re-enters the legitimate economy as apparently clean funds (e.g., purchasing real estate, paying salaries through controlled companies, making investments).

The *layering* stage is the primary target of graph-based detection: it generates characteristic multi-hop network patterns — cycles, fan-outs, rapid gather-then-scatter flows — that are invisible when examining individual transactions in isolation but become detectable when modelling the full transaction network.

### 1.2 Why Graphs and GNNs?

Financial transactions naturally form a directed graph: **nodes** represent bank accounts and **edges** represent transactions between them. Each edge carries transaction-level attributes (amount, payment format, currency, timestamp). The laundering signal does not reside in any single transaction; it emerges from the structural patterns formed by sequences of transactions.

Graph Neural Networks (GNNs) are a natural fit because they propagate information across the graph through message passing, enabling each transaction to be evaluated in the context of its full account neighbourhood. The specific task here is **edge classification**: for each transaction (edge), predict whether it is part of a laundering scheme (label 1) or legitimate (label 0).

### 1.3 Research Question

This project investigates a specific hypothesis: **can manual feature engineering compensate for the expressiveness ceiling imposed by the 1-Weisfeiler-Leman (1-WL) test on standard message-passing GNNs?**

Two complementary strategies are evaluated:

1. **Graph Feature Preprocessor (GFP):** pre-computed structural edge features using IBM SnapML that encode AML-specific graph patterns (cycles, scatter-gather, vertex statistics) directly into the edge attribute vector.
2. **Random Walk Positional Encoding (RWPE):** structural node-level positional encoding derived from random walk landing probabilities, injected into the node feature vector.

These are tested across three GNN architectures — **GINe**, **GATv2**, and **PNA** — in five configurations each: Baseline, GFP, RWPE, GFP+RWPE, and GFP with Split Projection.

---

## 2. GNNs and the 1-WL Expressiveness Limitation

### 2.1 The Weisfeiler-Leman Graph Isomorphism Test

The 1-WL test is a classical algorithm for checking whether two graphs are isomorphic. It works iteratively: each node is assigned a colour (initially based on its label or degree), and at each step a new colour is computed by hashing the multiset of colours of the node's neighbours together with its own colour. If at any iteration the colour histograms of the two graphs differ, they are declared non-isomorphic.

Formally, for a node $v$ at iteration $t$:

$$c^{(t)}(v) = \text{HASH}\left(c^{(t-1)}(v),\; \left\{\!\!\left\{ c^{(t-1)}(u) : u \in \mathcal{N}(v) \right\}\!\!\right\}\right)$$

### 2.2 The GNN–1-WL Equivalence

Xu et al. (2019) proved that any standard message-passing GNN is **at most as powerful** as the 1-WL test in distinguishing graph structures. Specifically, if two graphs cannot be distinguished by 1-WL, no standard GNN can distinguish them either, regardless of the number of layers, aggregation function, or update function. The message-passing update rule:

$$h_v^{(l)} = \text{UPDATE}^{(l)}\!\left(h_v^{(l-1)},\; \text{AGG}^{(l)}\!\left(\left\{\!\!\left\{ h_u^{(l-1)} : u \in \mathcal{N}(v) \right\}\!\!\right\}\right)\right)$$

is structurally equivalent to the 1-WL colour refinement: the hidden state at layer $l$ corresponds to the colour at iteration $t = l$.

### 2.3 What 1-WL Cannot Distinguish

The 1-WL test fails to distinguish many practically important structures:

- **Regular graphs** with the same degree sequence (e.g., two different 3-regular graphs)
- **Cycles of equal length** vs. collections of paths with the same total length
- **Structurally symmetric subgraphs** that appear locally identical even if the global structure differs

For AML detection, this is critical. The 8 canonical laundering patterns in the dataset include cycles (both short round-trips and long rings), fan-outs and fan-ins that can appear locally identical to legitimate hubs, and gather-scatter / scatter-gather patterns that exploit intermediate accounts. The 1-WL test treats all these as equivalent to locally similar but structurally different benign configurations.

### 2.4 Our Mitigation Strategy

Rather than replacing 1-WL-equivalent GNNs with provably more powerful architectures (which carry significant computational cost), we investigate whether **pre-computing the structural information that 1-WL misses** and injecting it as additional input features can bridge the performance gap. This approach is grounded in the observation that the 1-WL limitation affects *what the GNN can derive from message passing*, but does not constrain *what can be provided as input features*.

---

## 3. Dataset Description

### 3.1 IBM AML Benchmark — LI-Small

The dataset is the **LI-Small** (Low Illicit, Small) subset of the IBM Transactions for Anti-Money Laundering (AML) dataset, introduced by Altman et al. (2023). It is a realistic synthetic dataset generated by the AMLworld agent-based simulator, which models the complete money laundering cycle (placement, layering, and integration) in a virtual world of banks, individuals, and companies.

| Property | Value |
|---|---|
| Total Transactions (edges) | 6,924,049 |
| Bank Accounts (nodes, accounts file) | 712,688 |
| Unique Sender Accounts | 681,281 |
| Unique Receiver Accounts | 576,176 |
| Unique Accounts (union) | 705,903 |
| Laundering Rate | 0.0515% (3,565 transactions) |
| Class Ratio (Normal : Laundering) | 1,941:1 |
| Date Range | 2022-09-01 00:00 → 2022-09-17 15:28 |
| Effective High-Volume Period | September 1–10, 2022 (Sep 11–17: < 300 total transactions) |
| Payment Formats | 7 |
| Payment Currencies | 15 |

> **Important note on the date range:** The dataset spans 17 calendar days (Sep 1–17), but the transaction volume drops to nearly zero after September 10. From September 11 to 17 there are a combined total of only 223 transactions (77, 48, 48, 31, 9, 7, 3 per day respectively). This is an artefact of the synthetic generation process and means the **effective observation window is approximately 10 days**. The EDA and temporal features are primarily meaningful for September 1–10.

### 3.2 Raw Data Structure

**LI-Small_Trans.csv** (6,924,049 rows × 11 columns):

| Column | Type | Description |
|---|---|---|
| `Timestamp` | datetime | Transaction datetime (YYYY/MM/DD HH:MM) |
| `From Bank` | int | Numeric bank ID of sender |
| `Account` (1st) | string | Sender account identifier (hex string) |
| `To Bank` | int | Numeric bank ID of receiver |
| `Account` (2nd) | string | Receiver account identifier (hex string) |
| `Amount Received` | float | Amount in receiving currency |
| `Receiving Currency` | string | Currency received |
| `Amount Paid` | float | Amount in payment currency |
| `Payment Currency` | string | Currency paid |
| `Payment Format` | string | ACH / Bitcoin / Cash / Cheque / Credit Card / Reinvestment / Wire |
| `Is Laundering` | int | Binary label: 0 = legitimate, 1 = laundering |

**LI-Small_accounts.csv** (712,688 rows × 5 columns):

| Column | Description |
|---|---|
| `Bank Name` | Human-readable bank name (e.g., "China Bank #2820") |
| `Bank ID` | Numeric bank identifier |
| `Account Number` | Account hex identifier (matches transaction columns) |
| `Entity ID` | Legal entity identifier |
| `Entity Name` | Entity name encoding type (e.g., "Corporation #41344", "Individual #23001") |

**LI-Small_Patterns.txt:** Ground-truth annotations file containing 117 laundering attempts with `BEGIN/END LAUNDERING ATTEMPT - <TYPE>` markers, across 8 base pattern types (44 distinct parameterised variants).

### 3.3 Laundering Pattern Typologies

The dataset contains exactly **8 base pattern types** as defined by Altman et al. (2023), with 117 total attempts and 44 distinct parameterised variants (differing in degree, hop count, etc.). There is **no "integration" pattern label** in the patterns file — integration is part of the AMLworld simulation framework's conceptual model, but the labelled laundering transactions in the dataset all correspond to the 8 structural patterns below.

| Pattern | Count in LI-Small | Graph Shape | AML Description |
|---|---|---|---|
| **Stack** | 18 | Layered bipartite | Bipartite + intermediate layer; multi-hop fund movement through structured account layers |
| **Bipartite** | 16 | Bipartite | Funds move from a set of input accounts to a set of output accounts simultaneously |
| **Scatter-Gather** | 13 | Bipartite hub | Fan-out from v to intermediate nodes, then fan-in from those same nodes to u |
| **Gather-Scatter** | 9 | Hub | Fan-in then fan-out at the same vertex (consolidate then distribute) |
| **Fan-Out** | 17 | Star (1→many) | Single account distributes to many recipients (smurfing) |
| **Fan-In** | 12 | Star (many→1) | Many accounts consolidate into one |
| **Cycle** | 11 | Ring | Funds circulate back to the originating account (round-trip laundering) |
| **Random** | 17 | Random walk | Multi-hop chain without returning to origin; resembles legitimate transaction chains |

> **Note:** Stack (18) and Bipartite (16) are the most frequent, both requiring at least 2-hop message passing to detect. Scatter-Gather (13) is the third most common. The Fan-Out (17 total), Fan-In (12 total), Cycle (11 total), and Random (17 total) counts are obtained by summing over their parameterised variants.

The **44 distinct parameterised variants** encode the degree and hop parameters, e.g.: "Fan-Out: Max 7-degree Fan-Out", "Cycle: Max 11 hops", "Gather-Scatter: Max 16-degree Fan-In". This means the GNN must generalise across different hub sizes rather than memorising a fixed topology.

---

## 4. Exploratory Data Analysis

### 4.1 Class Imbalance

The dataset exhibits **severe class imbalance**:

- Total transactions: 6,924,049
- Laundering transactions: 3,566 (0.0515%)
- Legitimate transactions: 6,920,483 (99.9485%)
- Class ratio: **1,941:1** (overall dataset); train split specifically is ~2,290:1

This is one of the central challenges: a naive classifier predicting "always legitimate" achieves 99.95% accuracy but catches zero launderers. Standard accuracy is therefore meaningless, and specialised metrics (PR-AUC, F1, MCC) are required.

### 4.2 Transaction Volume Distribution Over Time

The temporal distribution of transactions reveals an important characteristic of the dataset:

| Date | Transaction Count |
|---|---|
| 2022-09-01 | 1,524,807 |
| 2022-09-02 | 1,027,758 |
| 2022-09-03 | 283,326 |
| 2022-09-04 | 282,476 |
| 2022-09-05 | 657,397 |
| 2022-09-06 | 657,170 |
| 2022-09-07 | 658,504 |
| 2022-09-08 | 657,938 |
| 2022-09-09 | 891,573 |
| 2022-09-10 | 282,877 |
| 2022-09-11–17 | 223 total |

September 1 accounts for 22% of all transactions alone. The average across the active 10-day window is approximately 407,297 transactions/day; laundering averages 209.7 transactions/day. After September 10, transaction volume essentially stops — this is a synthetic generation artefact and is not representative of real-world behaviour. The training/validation/test split (60/20/20 by time) is therefore dominated by September 1–10 data.

### 4.3 Temporal Laundering Patterns

Analysis of laundering rates by time reveals:

- **Hour of day:** Laundering rate is lowest around midnight (~0.015%) and peaks during midday hours 12:00–16:00 (~0.07%). The peak hour is **13:00**, suggesting laundering activity in this dataset follows business hours.
- **Day of week:** Weekend days (Saturday and Sunday) show **significantly elevated laundering rates** (~0.10–0.12% vs. ~0.05% average). The day with the highest laundering rate is **Sunday**. This ~2× weekend effect is used as the `Is_Weekend` binary feature.

Cyclical encoding (`sin`/`cos`) is used for temporal features to avoid the artificial discontinuity at midnight and end-of-week boundaries.

### 4.4 Transaction Amount Distribution

Key statistics from the EDA:

| Percentile | Normal ($) | Laundering ($) |
|---|---|---|
| p25 | 175.22 | 1,461.77 |
| p50 | 1,398.16 | 4,295.24 |
| p75 | 12,224.89 | 13,971.04 |
| p90 | 131,117.04 | 60,990.33 |
| p95 | 600,226.34 | 312,362.35 |
| p99 | 12,359,560.31 | 21,641,690.30 |
| Median | $1,399.44 | — |
| Mean | $4,676,035.97 | — |
| Max | $3,644,853,662,746.95 | — |

The distribution is **extremely heavy-tailed** (max = ~$3.6 trillion), making `log(1 + amount)` transformation essential. Laundering transactions show a distribution concentrated at mid-range values ($1K–$8K), but there is no clean threshold separating them from legitimate transactions — amount alone is insufficient for detection.

**Currency mismatch rate:** 1.43% of all transactions involve currency conversion (Amount Paid currency ≠ Amount Received currency). Crucially, in this synthetic dataset, **cross-currency transactions have zero laundering** — a synthetic artefact worth noting. The `Currency_Mismatch` binary feature is retained as a structural signal despite this.

### 4.5 Payment Format Analysis

The payment format is the strongest single predictor of laundering risk in this dataset:

| Payment Format | Count | Laund. Count | Laund. Rate | Share (%) | Relative Risk |
|---|---|---|---|---|---|
| ACH | 796,581 | 2,611 | **0.33%** | 11.50 | **6.4× avg** |
| Bitcoin | 309,208 | 110 | 0.036% | 4.47 | 0.7× avg |
| Cash | 655,688 | 124 | 0.019% | 9.47 | 0.4× avg |
| Cheque | 2,503,158 | 459 | 0.018% | 36.15 | 0.35× avg |
| Credit Card | 1,780,389 | 261 | 0.015% | 25.71 | 0.29× avg |
| Reinvestment | 650,458 | **0** | 0.000% | 9.39 | 0× avg |
| Wire | 228,567 | **0** | 0.000% | 3.30 | 0× avg |

Key observations:
- **ACH**: despite representing only 11.5% of transactions, carries a laundering rate of **0.33%** — approximately **6.4× the overall average**. This makes `Is_ACH` the single most informative binary feature.
- **Reinvestment and Wire** have **zero laundering transactions** in this dataset. Reinvestment transactions are intra-account bookkeeping entries (self-loops in the graph sense).
- **Bitcoin** has a slightly below-average laundering rate (0.036%), contrary to real-world intuition — a synthetic dataset limitation.
- **Cash** also shows below-average risk in this dataset, contrasting with real-world placement-stage concerns.

### 4.6 Self-Loop Transactions

Transactions where the sender and receiver accounts are identical:

- Self-loop count: **804,477 (11.62% of all transactions)**
- Self-loop laundering rate: **0.000004%** (essentially zero — only 3 laundering self-loops in the entire dataset)
- Format breakdown: Reinvestment (650,458), ACH (96,141), Bitcoin (44,982), Cheque (6,341), Credit Card (4,308), Cash (1,471), Wire (776)

The near-zero laundering rate for self-loops, combined with their 11.62% share of all transactions, means the `Is_Self_Loop` binary feature allows the model to confidently classify this large subpopulation as legitimate.

### 4.7 Network / Graph Structure Analysis

The financial network exhibits scale-free properties characteristic of real financial transaction graphs:

| Metric | Value |
|---|---|
| Out-Degree: mean | 14.7 |
| Out-Degree: median | 2 |
| Out-Degree: p95 | 57 |
| Out-Degree: max | 222,037 |
| In-Degree: mean | 15.8 |
| In-Degree: median | 16 |
| In-Degree: p95 | 39 |
| In-Degree: max | 1,553 |

The extreme skew in out-degree (mean 14.7, max 222,037) reflects the scale-free nature of the network — a tiny fraction of accounts are responsible for enormous transaction volumes. The top active senders have out-degrees of 222,037; 138,777; 42,385 etc., but their laundering rates are actually below-average (0.0014%), consistent with these being legitimate high-volume financial institutions rather than criminal hubs.

### 4.8 Entity Type Analysis

| Entity Type | Transaction Count | Laundering Count | Laundering Rate |
|---|---|---|---|
| Partnership | 2,484,005 | 1,288 | 0.0518% |
| Sole (Proprietorship) | 2,378,575 | 970 | 0.0408% |
| Corporation | 2,048,245 | 1,292 | 0.0631% |
| Individual | 13,249 | 15 | **0.1132%** |

Individual accounts show the highest laundering rate (~**2.2× the overall average**), consistent with AML intuition that individual accounts are often used in placement or integration stages. However, their small transaction count (13,249, less than 0.2% of all transactions) limits their aggregate contribution.

### 4.9 Currency Analysis

The dataset contains **15 distinct payment currencies** and 15 receiving currencies. Despite this diversity:
- 1.43% of transactions involve currency conversion (mismatch between payment and receiving currency)
- **Zero laundering transactions occur in cross-currency transactions** in this dataset

This is a known synthetic dataset limitation. In real-world AML, currency conversion is a significant layering signal. The `Currency_Mismatch` feature is retained as a structural flag for real-world generalisability.

---

## 5. Data Preparation and Feature Engineering (Baseline)

### 5.1 Design Philosophy

The baseline feature engineering follows a strict **no-leakage** principle: all features must be derivable from information available at the time of each transaction, using only account metadata and the transaction's own attributes. The graph task is **edge classification** with a cumulative snapshot design:

- `train_graph`: edges from the first 60% of the time window (≈ Sep 1–7, 4.15M edges)
- `val_graph`: `train_graph` edges + the next 20% (1.38M new eval edges; prior edges provide neighbourhood context)
- `test_graph`: all edges (1.38M new test edges evaluated; all prior edges provide context)

This cumulative design ensures that suspicious test-period transactions can be evaluated with full knowledge of both accounts' prior transaction history during message passing.

### 5.2 Edge Feature Engineering (16 dimensions)

| Feature | Dim | Description | AML Signal |
|---|---|---|---|
| `Amount_Log` | 1 | log1p(Amount Paid) — compresses heavy tail | Transaction magnitude |
| `Currency_Mismatch` | 1 | Receiving currency ≠ Payment currency | Structural flag |
| `Hour_Sin` | 1 | sin(2π × hour / 24) | Cyclical hour encoding |
| `Hour_Cos` | 1 | cos(2π × hour / 24) | Cyclical hour encoding |
| `DayOfWeek_Sin` | 1 | sin(2π × dow / 7) | Cyclical day encoding |
| `DayOfWeek_Cos` | 1 | cos(2π × dow / 7) | Cyclical day encoding |
| `Is_Weekend` | 1 | Saturday or Sunday = 1 | ~2× laundering rate |
| `Is_ACH` | 1 | Payment format == ACH | ~6.4× laundering rate |
| `Is_Self_Loop` | 1 | src_account == dst_account | Near-zero risk |
| `PayFmt_ACH` | 1 | One-hot: ACH | |
| `PayFmt_Bitcoin` | 1 | One-hot: Bitcoin | |
| `PayFmt_Cheque` | 1 | One-hot: Cheque | |
| `PayFmt_CreditCard` | 1 | One-hot: Credit Card | |
| `PayFmt_Cash` | 1 | One-hot: Cash | |
| `PayFmt_Reinvestment` | 1 | One-hot: Reinvestment | |
| `PayFmt_Wire` | 1 | One-hot: Wire | |

**Total baseline edge feature dimension: 16**

Cyclical encoding eliminates the artificial discontinuity between hour 23 and hour 0 (one hour apart in reality, but far apart as integer values). One-hot encoding for all 7 payment formats is preferred over ordinal encoding since there is no natural ordering between ACH, Wire, Cheque, etc.

### 5.3 Node Feature Engineering (5 dimensions)

Node features are derived purely from the accounts table:

| Feature | Dim | Description |
|---|---|---|
| `Bank_ID_Norm` | 1 | Bank ID normalised to [0,1]: (id − min) / (max − min + ε) |
| `EntType_Corporation` | 1 | One-hot entity type |
| `EntType_Individual` | 1 | One-hot entity type |
| `EntType_Partnership` | 1 | One-hot entity type |
| `EntType_Sole` | 1 | One-hot entity type (Sole Proprietorship) |

**Total baseline node feature dimension: 5**

Account identifiers (hex strings) are explicitly excluded as features, following the paper's methodology, to prevent the model from memorising account identities rather than learning laundering patterns.

### 5.4 Graph Split Statistics

| Split | Total Edges | Eval Edges | Laundering | Rate |
|---|---|---|---|---|
| Train | 4,154,429 | 4,154,429 | 1,813 | 0.0436% |
| Validation | 5,539,239 | 1,384,810 | 827 | 0.0597% |
| Test | 6,924,049 | 1,384,810 | 925 | 0.0668% |
| Train class ratio | — | — | — | 2,290:1 |

The slightly higher laundering rate in validation and test compared to training reflects temporal drift in the synthetic data — laundering activity intensifies during the later portion of the observation window.

---

## 6. Model Architecture

### 6.1 The Encode–Decode Pattern for Edge Classification

Standard GNNs produce **node embeddings** through message passing. Since the task is **edge classification**, an encode–decode structure is required:

1. **Encode:** Run message passing over the context graph to produce node embeddings $h_v$ for all relevant accounts.
2. **Decode:** For each seed transaction $(u \to v, e_{uv})$ to classify, concatenate the sender embedding, receiver embedding, and the transaction's own projected features, then pass through an MLP classifier.

PyG's `LinkNeighborLoader` maintains two distinct edge sets per batch:
- `batch.edge_index` — context edges used **only** for message passing (no direct labels used during this phase)
- `batch.edge_label_index` — seed edges to classify (have labels; gradients flow through these)

### 6.2 Encoder

```
node_proj:  x [N, node_dim]    → h0 [N, hidden_dim=64]   via Linear + ReLU
edge_proj:  e [E, edge_dim]    → e0 [E, hidden_dim=64]   via Linear + ReLU
```

The edge projection is **shared between the encoder and decoder**: the same `edge_proj` linear layer is reused to project both context edges (during message passing) and seed edges (during classification). This weight sharing ensures that edge features are always embedded in the same 64-dimensional space regardless of whether they appear as context or classification targets.

### 6.3 Message Passing Layers

All three architectures use **2 GNN layers**, `hidden_dim = 64`, `dropout = 0.3`, and `LayerNorm` after each layer. The critical shared property is that **all three architectures use edge features in every message passing layer** — not only in the decoder. Edge features modulate each message, so transaction-level signals (amount, format, timing) influence what each account "sends" to its neighbours. Without edge features in message passing, all transactions between the same pair of accounts would produce identical messages regardless of their amount or format.

**Why LayerNorm over BatchNorm:** With a 2,290:1 (train) class imbalance, BatchNorm statistics are dominated almost entirely by legitimate transactions. LayerNorm normalises each node independently across its 64 feature dimensions, preserving laundering-specific activation patterns that would otherwise be washed out by the majority-class mean and variance.

#### 6.3.1 GINe (Graph Isomorphism Network with Edge Features)

GINe extends GIN (Xu et al., 2019) by incorporating edge features into the message computation. The aggregation at layer $l$ for node $v$ is:

$$h_v^{(l)} = \text{MLP}^{(l)}\!\left((1+\varepsilon^{(l)}) \cdot h_v^{(l-1)} + \sum_{u \in \mathcal{N}(v)} \text{ReLU}\!\left(h_u^{(l-1)} + e_{uv}^{(l-1)}\right)\right)$$

where:
- $h_v^{(l-1)} \in \mathbb{R}^{64}$ — current node embedding
- $e_{uv}^{(l-1)} \in \mathbb{R}^{64}$ — projected edge feature for the transaction $u \to v$ (baseline 16 dims projected to 64)
- $\varepsilon^{(l)}$ — learnable scalar (initialised to 0)
- $\text{MLP}^{(l)}$: Linear(64→128) → ReLU → LayerNorm(128) → Dropout(0.3) → Linear(128→64)

The $\text{ReLU}(h_u + e_{uv})$ term is what makes GINe edge-aware: each message from neighbour $u$ is **modulated by the specific transaction** connecting them. Two accounts can send completely different messages to the same receiver if their transactions differ (ACH vs. Reinvestment, large vs. small amount). After GINEConv, each layer applies:

```
h ← LayerNorm(h)   # per-node normalisation across 64 features
h ← ReLU(h)        # non-linearity
h ← Dropout(0.3)   # stochastic regularisation (training only)
```

**Why GINe is theoretically optimal among 1-WL-equivalent GNNs:** The sum aggregation + MLP combination is injective, making GINe as powerful as the 1-WL test — the strongest expressiveness achievable within the standard MPNN framework.

**Complete GINe forward pass:**

```
INPUT
  x           : [N, 5]         raw node features
  edge_attr   : [E, 16]        context edge features
  edge_label_index, edge_label_attr : seed edges

ENCODE
  h0 = ReLU(node_proj(x))            [N, 64]
  e0 = ReLU(edge_proj(edge_attr))    [E, 64]

  Layer 1 (GINEConv + edge features):
    agg[v] = Σ_{u∈N(v)} ReLU(h0[u] + e0[u→v])
    h_new[v] = MLP(h0[v] + agg[v])    # Linear(64→128)→ReLU→LN→Dropout→Linear(128→64)
    h1 = Dropout(ReLU(LayerNorm(h_new)))

  Layer 2 (GINEConv): identical structure using h1 and e0
    h2 = Dropout(ReLU(LayerNorm(MLP(h1[v] + Σ ReLU(h1[u] + e0[u→v])))))

DECODE
  e_seed = ReLU(edge_proj(edge_label_attr))     [n_seeds, 64]
  edge_emb = cat(h2[src], h2[dst], e_seed)      [n_seeds, 192]
  logit = edge_classifier(edge_emb)              [n_seeds]    # 192→64→1
```

#### 6.3.2 GATv2 (Graph Attention Network v2)

GATv2 (Brody et al., 2022) computes dynamic attention scores for each neighbour, with both source and target node features jointly contributing to the coefficient. The attention-weighted aggregation is:

$$h_v^{(l)} = \underset{k=1}{\overset{K}{\Big\|}} \sum_{u \in \mathcal{N}(v)} \alpha_{vu}^{(k)} \cdot W_k^{(l)} h_u^{(l-1)}$$

where the attention coefficient is:

$$\alpha_{vu}^{(k)} = \frac{\exp\!\left(\mathbf{a}_k^\top \cdot \text{LeakyReLU}\!\left(W_k^{(l)}\left[h_v^{(l-1)} \,\|\, h_u^{(l-1)} \,\|\, e_{vu}^{(l-1)}\right]\right)\right)}{\sum_{w \in \mathcal{N}(v)} \exp\!\left(\mathbf{a}_k^\top \cdot \text{LeakyReLU}\!\left(W_k^{(l)}\left[h_v^{(l-1)} \,\|\, h_w^{(l-1)} \,\|\, e_{vw}^{(l-1)}\right]\right)\right)}$$

with $K = 4$ attention heads concatenated: $4 \times (64/4) = 64$ output dimensions. `add_self_loops = False`.

**Edge features in the attention mechanism:** $e_{vu}^{(l-1)} \in \mathbb{R}^{64}$ (the projected 16-dim baseline features) is included in the concatenation that computes the attention score. This means the attention weight $\alpha_{vu}$ depends on the specific transaction connecting $u$ to $v$ — not just on the account embeddings. A high-value ACH transaction can receive a different (higher) attention weight than a low-value Reinvestment self-loop from the same sender.

**Key improvement over GAT v1:** In the original GAT, the attention coefficient is effectively static — it depends on a linear combination of $h_v$ and $h_u$ computed before the non-linearity. GATv2 applies the attention vector after the LeakyReLU, making the coefficient truly dynamic: the importance of neighbour $u$ to node $v$ genuinely depends on both nodes' current states.

#### 6.3.3 PNA (Principal Neighbourhood Aggregation)

PNA (Corso et al., 2020) addresses a different limitation: using a single aggregation function (e.g., mean) loses information because the same mean can result from completely different multisets of neighbour features. PNA combines **multiple aggregators** with **degree-based scalers**:

$$h_v^{(l)} = U\!\left(\bigoplus_{s \in \mathcal{S}}\bigoplus_{a \in \mathcal{A}} s\!\left(a\!\left(\left\{\!\!\left\{ \text{msg}(h_v^{(l-1)}, h_u^{(l-1)}, e_{vu}^{(l-1)}) : u \in \mathcal{N}(v) \right\}\!\!\right\}\right),\, d_v\right)\right)$$

where:
- $\mathcal{A} = \{\text{mean}, \text{min}, \text{max}, \text{std}\}$ — 4 aggregators
- $\mathcal{S} = \{\text{identity}, \text{amplification}, \text{attenuation}\}$ — 3 degree-based scalers
- $d_v$ = in-degree of node $v$; $\bigoplus$ = concatenation → 4 × 3 = **12** aggregated vectors

The three degree-based scalers:

$$s_{\text{identity}}(x, d) = x \qquad s_{\text{amplification}}(x, d) = x \cdot \frac{\log(d+1)}{\delta_{\max}} \qquad s_{\text{attenuation}}(x, d) = x \cdot \frac{\delta_{\max}}{\log(d+1)}$$

where $\delta_{\max} = \log(\overline{d}+1)$ and $\overline{d}$ is the average training-graph degree. Amplification boosts signals from high-degree hubs; attenuation suppresses them. This degree-awareness is critical for financial networks where hub accounts (high-volume money mules vs. legitimate clearinghouses) carry distinct structural significance.

**Edge features in PNA:** The message function $\text{msg}(h_v, h_u, e_{vu})$ incorporates the projected edge attributes $e_{vu} \in \mathbb{R}^{64}$. All four aggregators compute summaries over edge-feature-weighted messages, so transaction-level signals modulate each of the 12 aggregated vectors. The update network $U$ (implemented with `towers=4`) reduces the 12 × 64 = 768-dimensional concatenation back to 64.

**Degree histogram pre-computation (training graph only):**
```python
deg = degree(train_graph.edge_index[1], num_nodes=train_graph.num_nodes, dtype=torch.long)
deg = torch.bincount(deg, minlength=1)
# Used by PNAConv to calibrate δ_max
```

### 6.4 Decoder (All Architectures — Standard)

After the encoder produces node embeddings $h^{(L)} \in \mathbb{R}^{N \times 64}$, the decoder classifies each seed transaction:

$$\text{logit}_{uv} = \text{MLP}_{\text{cls}}\!\left(\left[h_u^{(L)} \,\|\, h_v^{(L)} \,\|\, \tilde{e}_{uv}\right]\right)$$

where $\tilde{e}_{uv} = \text{ReLU}(\text{edge\_proj}(e_{uv})) \in \mathbb{R}^{64}$ is the projected seed transaction features (same shared `edge_proj` used in the encoder).

Input dimension: $64 \times 3 = 192$. Classifier MLP:
```
Linear(192 → 64) → ReLU → LayerNorm(64) → Dropout(0.3) → Linear(64 → 1)
```

The output is a raw logit; $P(\text{laundering}) = \sigma(\text{logit})$ where $\sigma$ is the sigmoid function.

**Three information sources in the decoder:**
- `h[src]` — what kind of account the sender is (its 2-hop neighbourhood context from message passing)
- `h[dst]` — what kind of account the receiver is
- `e_seed` — the specific features of this particular transaction (amount, format, time of day)

Without `e_seed`, the model would assign the same laundering score to every transaction between the same pair of accounts regardless of amount, format, or timing — which is clearly wrong.

---

## 7. Training Setup

### 7.1 Mini-batch Sampling

PyG's `LinkNeighborLoader` is used with `num_neighbors = [100, 100]` (100 one-hop and 100 two-hop neighbours per seed edge endpoint, matching the paper's protocol). Each batch contains:
- `batch.edge_index` — context edges for message passing
- `batch.edge_label_index` — seed edges to classify

Seed edge features are tracked separately and looked up via `batch.input_id`:
```python
seed_attr = seed_edge_attr[batch.input_id.cpu()].to(device)
```

### 7.2 Loss Function and Class Imbalance Handling

$$\mathcal{L} = -\frac{1}{N} \sum_i \left[w \cdot y_i \log\sigma(\hat{y}_i) + (1-y_i)\log(1-\sigma(\hat{y}_i))\right]$$

The positive class weight $w$ is set to **50** for baseline models (reflecting the 2,290:1 train imbalance) and **8** for GFP models (which use more data per batch, reducing variance). Gradient clipping with `max_norm = 1.0` prevents exploding gradients from the sparse high-weight laundering signal.

### 7.3 Hyperparameters

| Parameter | Baseline | GFP / RWPE | GFP Split |
|---|---|---|---|
| Hidden dim | 64 | 64 | **128** |
| GNN layers | 2 | 2 | 2 |
| Dropout | 0.3 | 0.3 | 0.3 |
| Epochs | 10 | 20 | 20 |
| Batch size | 16,384 | 2,048–4,096 | 2,048 |
| Learning rate | 1e-3 | 1e-3 | 1e-3 |
| Weight decay | 1e-5 | 1e-5 | 1e-5 |
| Num neighbours | [100, 100] | [100, 100] | [100, 100] |
| pos_weight | 50 | 8 | 8 |
| Optimiser | Adam | Adam | Adam |
| LR scheduler | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |

### 7.4 Threshold Optimisation

Models are monitored at `threshold = 0.5` during training. After training, two optimal thresholds are found on the validation set:
- `thresh_F1`: maximises validation F1
- `thresh_MCC`: maximises validation MCC

These are then applied to the test set. Due to extreme class imbalance, the optimal threshold is typically well below 0.5.

### 7.5 Evaluation Metrics

**PR-AUC (Primary Metric):**
$$\text{PR-AUC} = \int_0^1 P(R)\, dR$$
Summarises classifier quality across all thresholds without inflation from the dominant negative class. A random classifier on a 0.05% positive-rate dataset achieves PR-AUC ≈ 0.0005; ROC-AUC ≈ 0.5.

**F1 Score:**
$$F_1 = \frac{2 \cdot \text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$
Balances precision (false positive cost = unnecessary investigations) against recall (false negative cost = missed crimes). Threshold-dependent; reported at the validation-optimised threshold.

**Matthews Correlation Coefficient (MCC):**
$$\text{MCC} = \frac{\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}$$
Balanced metric using all four confusion matrix cells. Range: $[-1, +1]$; 0 = random-chance. Particularly reliable for highly imbalanced binary classification.

---

## 8. Baseline Results

### 8.1 Comparison with Paper Benchmarks

The paper (Altman et al., 2023, Table 2) reports minority-class F1 scores on LI-Small with standard deviation across multiple seeds:

| Model | Paper F1 (%) ± std | Our F1 (%) | Alignment |
|---|---|---|---|
| GIN | 7.90 ± 2.78 | 9.62 | ✓ Within reported range |
| PNA | 16.45 ± 1.46 | 24.88 | Above paper (different hyperparams, threshold strategy) |
| GFP + XGBoost | 27.30 ± 0.33 | — | Baseline reference |

Our GINe aligns well with the paper's GIN result. PNA outperforms the paper's PNA — likely due to our validation-set threshold optimisation (paper uses fixed 0.5) and the longer 20-epoch training schedule.

### 8.2 Baseline Test Set Results

| Model | ROC-AUC (%) | PR-AUC (%) | F1 (%) | MCC |
|---|---|---|---|---|
| GINe Baseline | 96.18 | 3.46 | 9.62 | 13.02 |
| GAT Baseline | 95.06 | 2.15 | 6.54 | 8.36 |
| PNA Baseline | 96.64 | **21.73** | **24.88** | **30.54** |

**Key observations:**

1. **PNA baseline is dominant:** Despite using the same number of layers and hidden dimensions, PNA achieves PR-AUC = 21.73% vs. 3.46% (GINe) and 2.15% (GAT). This is because PNA's 4 aggregators × 3 scalers = 12 combined signals inherently capture richer neighbourhood statistics than single-aggregator GNNs. The degree-aware scalers also provide implicit structural position information.

2. **GAT baseline is weakest:** The attention mechanism — which in principle could learn to focus on suspicious transactions — fails with only baseline edge features. The 16-dimensional baseline features lack the structural pattern signals needed for attention weights to be meaningful discriminators.

3. **GINe baseline matches paper:** The 9.62% F1 and 13.02% MCC are consistent with the paper's reported 7.90 ± 2.78% F1 for GIN on LI-Small, validating the implementation.

4. **High ROC-AUC for all models** (95–97%): All models can rank laundering above legitimate on average, but the absolute separation is insufficient for accurate binary classification at any threshold — explaining the low F1 and PR-AUC values.

---

## 9. GFP Feature Engineering

### 9.1 Motivation

IBM SnapML's `GraphFeaturePreprocessor` (GFP) computes AML-specific structural features for every transaction in a single causal pass. For each edge $(u \to v, t)$, only edges with timestamp strictly less than $t$ are used — guaranteeing no temporal data leakage. The paper (Altman et al., Appendix D) describes the exact configuration used.

### 9.2 GFP Pattern Types and Configurations

#### 9.2.1 Scatter-Gather Histogram (3 dimensions)

- **Time window:** 6 hours (paper specification)
- **Bins:** [2–3), [3–5), [5–∞) gather/scatter nodes
- **Description:** For each transaction $(u \to v, t)$, counts distinct sources $u$ received from and distinct destinations $v$ forwarded to in a 6-hour window before $t$. Three histogram bins encode intensity.
- **AML signal:** The canonical "smurfing" or layering pattern — rapid gather-then-scatter. The [5–∞) bin is the most suspicious indicator.

#### 9.2.2 Temporal Cycle Histogram (3 dimensions)

- **Time window:** 24 hours (paper specification)
- **Bins:** [2–3) round-trips A→B→A, [3–5) short 3–4 node rings, [5–∞) complex rings
- **Description:** Counts distinct cycles involving this transaction that close within 24 hours.
- **AML signal:** Round-trips and short rings are hallmark laundering indicators. The 24-hour window captures rapid cycling characteristic of layering operations.

#### 9.2.3 Length-Constrained Cycle Histogram (3 dimensions)

- **Time window:** 24 hours
- **Maximum cycle length:** 6 (paper uses 10; our implementation uses 6 as a speed/memory compromise — bin boundary 5 < max_len 6 is maintained)
- **Bins:** [2–3), [3–5), [5–∞)
- **Description:** Counts simple cycles of length up to 6 involving this transaction that close within 24 hours.
- **AML signal:** Provides finer-grained cycle detection, capturing structurally simple short-path laundering circuits.

#### 9.2.4 Vertex Statistics (52 dimensions)

- **Time window:** 24 hours
- **Input columns:** Timestamp (col 3) and Amount_USD (col 4)
- **Statistics:** fan (0), degree (1), ratio (2), avg (3), sum (4), var (8), skew (9), kurtosis (10)
- **Directions:** Source node out-edges, source node in-edges, destination node out-edges, destination node in-edges
- **Total:** 8 statistics × 2 columns × 2 directions = 32 dims + degree/ratio per direction

Statistics capture account behavioural signatures in the preceding 24 hours:
- **fan:** distinct counterparties (= size of unique neighbourhood set)
- **degree:** total transaction count (= raw volume)
- **ratio:** fan/degree (counterparty uniqueness)
- **avg, sum:** mean/total amount or timestamp
- **var, skew, kurtosis:** higher-order moments — capture burst behaviour, e.g., sudden spike in transaction volume at unusual hours, or highly skewed amounts typical of structuring schemes

Note: fan/degree **histogram** bins are disabled (`fan=False`, `degree=False`) because vertex statistics already provide continuous values, which are strictly more informative than coarse histogram bins.

### 9.3 GFP Feature Summary

| Group | Config | Dims | Key AML Signal |
|---|---|---|---|
| Scatter-Gather | 6h window, bins [2,3,5] | 3 | Rapid gather-then-scatter |
| Temporal Cycle | 24h window, bins [2,3,5] | 3 | Cycle-closing transactions |
| LC-Cycle | 24h, max_len=6, bins [2,3,5] | 3 | Simple cycles up to length 6 |
| Vertex Stats (src out) | timestamp + Amount_USD, 8 stats | ~13 | Source account burst behaviour |
| Vertex Stats (src in) | timestamp + Amount_USD, 8 stats | ~13 | Source incoming patterns |
| Vertex Stats (dst out) | timestamp + Amount_USD, 8 stats | ~13 | Destination forwarding patterns |
| Vertex Stats (dst in) | timestamp + Amount_USD, 8 stats | ~13 | Destination receipt patterns |
| **Total GFP dims** | | **61** | |
| **Total edge dims (baseline + GFP)** | | **77** | |

### 9.4 FX Correction

In the GFP pipeline, `Amount_Log` is computed on **USD-converted amounts** using daily FX rates from Yahoo Finance (15 currencies in the dataset). Without FX correction, `Amount_Paid = 1000` means $1,000 for USD but potentially millions for Bitcoin, making amount-based vertex statistics incomparable across currencies.

### 9.5 GFP Feature Normalisation Pipeline

Fit on training data only, applied identically to val/test:

1. **Pattern histogram features** (scatter-gather, temp-cycle, lc-cycle): **no normalisation** — discrete count histograms already in bounded range.
2. **Log-scaled vertex stats** (fan, degree, sum, avg, var, kurtosis): `log1p → clip at [p01, p99] of training data → StandardScaler`.
3. **Standard-scaled vertex stats** (ratio, skew): `clip at [p01, p99] → StandardScaler` (no log, as these can be negative).

---

## 10. RWPE Positional Encoding

### 10.1 Concept and Motivation

Random Walk Positional Encoding (RWPE; Dwivedi et al., 2021) provides each node with a unique structural signature based on its local topology through random walk return probabilities. It directly addresses the 1-WL limitation at the node-feature level: two nodes that are structurally distinct (e.g., one in a 3-cycle and one in a 5-cycle) but indistinguishable by 1-WL will have different RWPE vectors.

### 10.2 RWPE Computation

For a directed graph with adjacency matrix $A$, define the column-stochastic transition matrix $\hat{A} = D^{-1}A$ where $D$ is the in-degree diagonal matrix. The RWPE vector for node $v$ is:

$$\text{RWPE}(v) = \left[(\hat{A}^1)_{vv},\; (\hat{A}^2)_{vv},\; \ldots,\; (\hat{A}^K)_{vv}\right] \in \mathbb{R}^K$$

where $(\hat{A}^k)_{vv}$ is the probability that a random walk of length $k$ starting at $v$ returns to $v$ after exactly $k$ steps.

**Configuration:** $K = 8$ walk steps. This captures cycles of length 2–8, covering round-trips (length 2), triangles (3), 4-cycles, and longer multi-hop laundering rings.

```python
from torch_geometric.transforms import AddRandomWalkPE

def add_rwpe(graph, walk_length=8):
    transform = AddRandomWalkPE(walk_length=walk_length, attr_name='rwpe')
    graph = transform(graph)
    graph.x = torch.cat([graph.x, graph.rwpe], dim=-1)  # [N, 5] → [N, 13]
    del graph.rwpe
    return graph

# Applied per snapshot to avoid temporal leakage
train_graph = add_rwpe(train_graph)  # node_dim: 5 → 13
val_graph   = add_rwpe(val_graph)    # node_dim: 5 → 13
test_graph  = add_rwpe(test_graph)   # node_dim: 5 → 13
```

RWPE is computed independently per snapshot using **only that snapshot's own edges** — temporal leakage is prevented because each snapshot's RWPE reflects only the transactions visible up to that point.

### 10.3 Structural Interpretation for AML

| Walk length $k$ | Structural meaning | AML relevance |
|---|---|---|
| $k=2$ | Round-trips: $A \to B \to A$ | Direct cash-back patterns; A→B→A is the simplest cycle |
| $k=3$ | Triangles: $A \to B \to C \to A$ | Three-account cycling rings |
| $k=4$ | 4-cycles and squares | Longer laundering rings through 3 intermediaries |
| $k=5,6$ | Pentagon and hexagon cycles | Multi-layer shell company structures |
| $k=7,8$ | Extended cycles | Complex cross-jurisdiction laundering chains |

Key structural properties encoded by RWPE:
- **High-degree hub accounts** (potential money mules or legitimate clearinghouses) have distinctive return probability profiles reflecting their connectivity
- **Cycle participants** have elevated $p_{vv}^{(k)}$ for the length $k$ of the cycle they belong to
- **Peripheral accounts** (appear in few transactions) have near-zero return probabilities at all steps

RWPE addresses **node-level structural identity** — complementary to GFP which operates at the **edge-level pattern** scale.

### 10.4 Architecture Change

When RWPE is added, the only architectural change is the node input dimension:
```
Baseline:  node_proj = Linear(5  → 64)
RWPE:      node_proj = Linear(13 → 64)   # 5 original + 8 RWPE
```
All other components — edge features, message passing, decoder — remain identical. RWPE is a drop-in enhancement.

---

## 11. GFP Split Projection Architecture

### 11.1 Motivation

In the standard GFP architecture, all 77 edge features (16 baseline + 61 GFP) are projected through the shared encoder before entering message passing:
```
e0 = ReLU(edge_proj(edge_attr_77))   →  [E, 64]
```

**The problem:** GFP features are pre-computed structural summaries — they already encode, as explicit scalars, the exact patterns the 1-WL GNN cannot derive. Passing them through the same encoder and message passing aggregation as transaction-level baseline features may dilute their discriminative signal. The GNN's aggregation is designed to build neighbourhood representations from raw features; applying it to already-summarised structural features may not be optimal.

### 11.2 Split Projection Design

The Split Projection architecture (called "Enhanced" in the codebase) separates the two types of edge features:

**Encoder (message passing):** Uses only the **16 baseline features**
```python
e_base = ReLU(edge_proj_base(edge_attr[:, :16]))  # [E, hidden_dim]
# GNN processes neighbourhood structure from raw transaction features only
```

**Decoder (classification):** Uses **both baseline and GFP features** through separate projections
```python
e_base = ReLU(edge_proj_base(edge_label_attr[:, :16]))        # [n_seeds, hidden_dim]
e_gfp  = ReLU(edge_proj_gfp(edge_label_attr[:, 16:]))         # [n_seeds, hidden_dim]
edge_emb = cat(h[src], h[dst], e_base, e_gfp)                 # [n_seeds, 4×hidden_dim]
logit = edge_classifier(edge_emb)                              # [n_seeds]
```

This decomposition creates an ensemble-like structure: the GNN provides learned structural embeddings from transaction context, and GFP provides pre-computed structural pattern scores. Both contribute to the final decision but through independent pathways.

### 11.3 Architecture Comparison

| Component | Baseline | Standard GFP | GFP Split |
|---|---|---|---|
| `node_proj` | Linear(5 → 64) | Linear(5 → 64) | Linear(5 → **128**) |
| `edge_proj` (encoder) | Linear(16 → 64) | Linear(77 → 64) | Linear(16 → **128**) |
| `edge_proj_gfp` (decoder only) | — | — | Linear(61 → **128**) |
| Hidden dim | 64 | 64 | **128** |
| Decoder input | 3 × 64 = 192 | 3 × 64 = 192 | **4 × 128 = 512** |
| Decoder MLP | 192→64→1 | 192→64→1 | **512→128→1** |

The hidden dimension is doubled to 128 in the split variant to provide sufficient capacity for the 4-component decoder. The `edge_proj_base` (Linear 16 → 128) is shared between encoder and decoder exactly as in the standard architecture.

### 11.4 PyTorch Implementation

```python
class GINe(nn.Module):
    def __init__(self, node_dim, base_edge_dim, gfp_edge_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.base_edge_dim = base_edge_dim  # 16

        self.node_proj      = nn.Linear(node_dim,     hidden_dim)   # 5 → 128
        self.edge_proj_base = nn.Linear(base_edge_dim, hidden_dim)  # 16 → 128 (shared)
        self.edge_proj_gfp  = nn.Linear(gfp_edge_dim,  hidden_dim)  # 61 → 128 (decode only)

        for _ in range(num_layers):
            mlp = build_mlp(hidden_dim, hidden_dim * 2, hidden_dim, dropout=dropout)
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.edge_classifier = build_mlp(hidden_dim * 4, hidden_dim, 1, dropout=dropout)

    def encode(self, x, edge_index, edge_attr):
        # Only baseline 16 features enter message passing
        e_base = F.relu(self.edge_proj_base(edge_attr[:, :self.base_edge_dim]))
        h = F.relu(self.node_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, e_base)  # GINEConv uses edge features
            h = F.dropout(F.relu(norm(h)), p=self.dropout, training=self.training)
        return h

    def decode(self, h, edge_label_index, edge_label_attr):
        src, dst = edge_label_index
        e_base = F.relu(self.edge_proj_base(edge_label_attr[:, :self.base_edge_dim]))
        e_gfp  = F.relu(self.edge_proj_gfp(edge_label_attr[:, self.base_edge_dim:]))
        edge_emb = torch.cat([h[src], h[dst], e_base, e_gfp], dim=-1)  # [n_seeds, 512]
        return self.edge_classifier(edge_emb).squeeze(-1)
```

---

## 12. Results: Full Comparison

### 12.1 Complete Test Set Results

Metrics reported on the test set using thresholds optimised on the validation set.

| Model | ROC-AUC (%) | PR-AUC (%) | F1 (%) | MCC |
|---|---|---|---|---|
| **GINe Baseline** | 96.18 | 3.46 | 9.62 | 13.02 |
| **GINe + GFP** | 97.17 | 18.56 | 23.88 | 27.02 |
| **GINe + RWPE** | 96.68 | 15.17 | 16.11 | 18.18 |
| **GINe + GFP+RWPE** | 97.08 | 19.65 | 24.46 | 26.70 |
| **GINe + GFP Split** | **97.64** | **22.73** | **26.24** | **31.54** |
| **GAT Baseline** | 95.06 | 2.15 | 6.54 | 8.36 |
| **GAT + GFP** | 96.83 | 18.38 | 22.47 | 26.44 |
| **GAT + RWPE** | 95.83 | 4.09 | 9.72 | 11.07 |
| **GAT + GFP+RWPE** | 96.85 | 17.70 | 22.05 | 26.41 |
| **GAT + GFP Split** | **97.21** | **21.91** | **24.87** | **30.35** |
| **PNA Baseline** | 96.64 | 21.73 | 24.88 | 30.54 |
| **PNA + GFP** | 96.78 | 18.49 | 23.65 | 26.87 |
| **PNA + RWPE** | **97.05** | **22.94** | **25.92** | **34.49** |
| **PNA + GFP+RWPE** | 96.98 | 19.89 | 25.71 | 29.10 |
| **PNA + GFP Split** | 97.13 | 21.27 | 24.66 | 31.28 |

Bold entries indicate the best-performing configuration per architecture.

### 12.2 Feature Configuration Impact

Absolute improvement over baseline, per architecture:

| Config | GINe ΔPR-AUC | GAT ΔPR-AUC | PNA ΔPR-AUC | GINe ΔMCC | GAT ΔMCC | PNA ΔMCC |
|---|---|---|---|---|---|---|
| + GFP | +15.10 | +16.23 | **-3.24** | +14.00 | +18.08 | **-3.67** |
| + RWPE | +11.71 | +1.94 | +1.21 | +5.16 | +2.71 | **+3.95** |
| + GFP+RWPE | +16.19 | +15.55 | **-1.84** | +13.68 | +18.05 | **-1.44** |
| + GFP Split | **+19.27** | **+19.76** | -0.46 | **+18.52** | **+21.99** | +0.74 |

### 12.3 Overall Best Results

| Metric | Best Model | Value |
|---|---|---|
| Highest PR-AUC | PNA + RWPE | 22.94% |
| Highest F1 | GINe + GFP Split | 26.24% |
| Highest MCC | **PNA + RWPE** | **34.49** |
| Best overall balance | GINe + GFP Split | PR-AUC 22.73%, F1 26.24%, MCC 31.54 |

---

## 13. Analysis and Discussion

### 13.1 Does Feature Engineering Mitigate 1-WL Limitations?

**Yes, substantially — but the effectiveness depends on the base architecture.**

For GINe and GAT (single-aggregator architectures with the lowest baselines), GFP features provide dramatic improvements. The structural patterns GINe cannot derive from message passing (cycles, scatter-gather, burst statistics) are directly provided as pre-computed features, and the models effectively learn to use them as discriminating signals.

For PNA (multi-aggregator, degree-aware architecture with a strong baseline), the picture is more nuanced:
- Standard GFP **decreases** PNA's performance — the 61 GFP dimensions mixed into the encoder may dilute PNA's carefully balanced multi-aggregator signals.
- RWPE **significantly improves** PNA (+3.95 MCC), suggesting PNA's degree-aware scalers synergise with node-level positional information.
- GFP Split Projection has minimal effect on PNA (−0.46 pp PR-AUC), suggesting PNA's inherent expressiveness partially substitutes for what GFP provides to GINe/GAT.

### 13.2 Why GFP Split Projection Outperforms Standard GFP for GINe and GAT

The split projection is a principled separation of two complementary functions:
1. **GNN encoder:** Learns to build account embeddings from raw transaction-level features through neighbourhood aggregation. Using only 16 baseline features keeps this signal clean.
2. **GFP in decoder:** Pre-computed structural summaries (cycle scores, scatter-gather counts, vertex statistics) are injected directly at the classification stage, bypassing aggregation entirely.

This prevents the GFP's already-summarised structural information from being re-processed through aggregation steps that are designed for raw feature transformation, not structural summary combination.

### 13.3 Why RWPE Synergises with PNA Specifically

PNA's degree-aware scalers already partially encode structural node roles (high-degree hubs are amplified/attenuated differently from low-degree nodes). RWPE provides a richer and more precise version of this positional information — exact cycle membership and structural identity rather than just degree counts. PNA's diverse aggregators are well-positioned to leverage this richer node encoding compared to the single-aggregator GINe and GAT.

For GINe and GAT, the primary bottleneck is the absence of explicit structural pattern features at the edge level (cycles, scatter-gather), which GFP addresses. RWPE's node-level encoding provides secondary but still significant gains (+11.71 pp PR-AUC for GINe alone).

### 13.4 Interpretation in Terms of Laundering Patterns

Given the dataset's 8 pattern types (dominated by Stack, Bipartite, Scatter-Gather, Fan-Out, Random):
- **GFP cycle features** directly encode the Cycle pattern and partially capture the Scatter-Gather pattern
- **GFP scatter-gather histogram** directly targets the Scatter-Gather and Gather-Scatter patterns
- **GFP vertex statistics** capture the burst behaviour characteristic of Fan-In and Fan-Out patterns
- **RWPE** helps detect cycle participants (elevated $p_{vv}^{(k)}$ for the cycle's length $k$) and hub accounts

The Stack and Bipartite patterns — the most frequent in the dataset — require multi-hop detection (minimum 2 GNN layers) and are more challenging because they do not always form closed cycles. Their detection relies more on the combination of GNN structural embeddings with vertex statistics from GFP.

### 13.5 Limitations

1. **17-day synthetic window:** The effective data is only Sep 1–10. Sep 11–17 contributes negligibly (~223 transactions). All temporal patterns are measured within a 10-day window, which limits assessment of longer-term laundering strategies.
2. **Synthetic data:** LI-Small is realistic but synthetic. Performance may differ on real AML data due to different laundering strategies, data quality, and network topology.
3. **Untested combination:** GFP Split + RWPE was not evaluated.
4. **lc-cycle max_len = 6 vs. paper's 10:** Some longer laundering cycles (length 7–10) are not captured by our GFP configuration.
5. **Zero laundering in cross-currency transactions:** The currency mismatch feature has zero predictive signal in this dataset, which contradicts real-world AML intuition.
6. **Wire and Reinvestment have zero laundering:** These two formats are entirely clean in this dataset, making their one-hot features uninformative — a potential over-feature-engineering issue.

---

## 14. Future Work

### 14.1 Immediate: GFP Split + RWPE

The most promising untested combination: **split projection with RWPE node encoding**. Specifically:
- **Encoder:** baseline 16 features + RWPE node embeddings (13-dim nodes)
- **Decoder:** baseline 16 + GFP 61 features injected separately

This combines the node structural identity from RWPE (which helps PNA most) with the clean encoder + decoder-injected GFP (which helps GINe/GAT most). Expected to be the strongest overall configuration based on the complementarity observed.

### 14.2 Full GFP Configuration

Re-running GFP with `lc-cycle_len = 10` (paper default) would capture longer laundering cycles (lengths 7–10) that may span more jurisdictions. The additional computational cost may be worthwhile given GFP's demonstrated impact.

### 14.3 Higher-Order GNNs

Testing provably more expressive architectures (OSAN, NGNN, k-WL GNNs) would provide a theoretical upper bound. If they significantly outperform the best feature-engineered 1-WL models, it implies there are laundering structures that feature engineering cannot adequately represent.

### 14.4 Temporal GNN Extensions

Architectures like TGN (Temporal Graph Networks) or TGAT model time-evolving graphs explicitly. Given that laundering patterns unfold over specific time windows (the GFP uses 6-hour and 24-hour windows), explicit temporal modelling in the GNN may complement the GFP's causal feature computation.

### 14.5 Ensemble of Best Models

PNA+RWPE (best MCC: 34.49) and GINe+GFP Split (best F1: 26.24%) have complementary strengths. An ensemble combining their predictions could achieve a superior precision-recall balance compared to any individual model.

---

## 15. Conclusions

This project evaluated whether manual feature engineering can mitigate the 1-WL expressiveness ceiling of standard GNNs for Anti-Money Laundering detection on the IBM LI-Small synthetic financial transaction dataset (6.9M transactions, 17-day period Sep 1–17 2022, with ~10 active days, 1,941:1 class imbalance, 8 base laundering pattern types).

**Core findings:**

1. **Feature engineering substantially compensates for 1-WL gaps in GINe and GAT.** GFP features increase PR-AUC by 15–20 percentage points for both architectures. The structural patterns these models cannot derive from message passing — cycles, scatter-gather, vertex burst statistics — are directly provided as pre-computed edge features.

2. **Split projection is a superior GFP integration strategy for GINe and GAT.** Injecting GFP features directly into the decoder (bypassing the encoder) outperforms encoder-integrated GFP across all metrics. The best GINe result (PR-AUC 22.73%, MCC 31.54) and best GAT result (PR-AUC 21.91%, MCC 30.35) are both from the split projection variants.

3. **RWPE provides complementary node-structural information.** It improves GINe by +11.71 pp PR-AUC alone, and critically improves PNA's MCC by +3.95 (to 34.49 — the strongest class separability across all experiments). The synergy between RWPE and PNA's degree-aware scalers is particularly notable.

4. **PNA's multi-aggregator design is inherently more expressive.** The PNA baseline (PR-AUC 21.73%, MCC 30.54) outperforms both GINe and GAT baselines without any additional features, demonstrating that architecture design choices have large impact even within the 1-WL equivalence class.

5. **Standard GFP hurts PNA.** Adding GFP to the PNA encoder decreases PR-AUC by 3.24 pp — the GFP features likely interfere with PNA's multi-aggregator balance. This underscores that feature engineering strategies must be tailored to the base architecture.

6. **Untested combination (GFP Split + RWPE) is the most promising future direction** and is expected to combine the best of both approaches for all three architectures.

**Research conclusion:** Manual feature engineering *can* partially bridge the 1-WL expressiveness gap, but the effectiveness is architecture-dependent. For information-poor single-aggregator GNNs (GINe, GAT), GFP's explicit structural patterns provide dramatic improvements. For the richer PNA architecture, RWPE's structural node identity encoding is more beneficial. The most impactful single architectural insight is the split projection design, which preserves GFP's discriminative precision by keeping it out of the message-passing aggregation pipeline.

---

## References

1. Altman, E., et al. (2023). *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models.* NeurIPS 2023 Datasets & Benchmarks. arXiv:2306.16424v3.

2. Xu, K., et al. (2019). *How Powerful are Graph Neural Networks?* ICLR 2019. arXiv:1810.00826.

3. Brody, S., Alon, U., & Yahav, E. (2022). *How Attentive are Graph Attention Networks?* ICLR 2022. arXiv:2105.14491.

4. Corso, G., et al. (2020). *Principal Neighbourhood Aggregation for Graph Nets.* NeurIPS 2020. arXiv:2004.05718.

5. Dwivedi, V. P., et al. (2021). *Graph Neural Networks with Learnable Structural and Positional Representations.* ICLR 2022. arXiv:2110.07875.

6. IBM Research. *Graph Feature Preprocessor (SnapML).* [https://snapml.readthedocs.io/en/latest/graph_preprocessor.html](https://snapml.readthedocs.io/en/latest/graph_preprocessor.html)

7. Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric.* arXiv:1903.02428.

8. PyTorch Geometric. *GINEConv.* [https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.nn.conv.GINEConv.html](https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.nn.conv.GINEConv.html)

9. PyTorch Geometric. *GATv2Conv.* [https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.GATv2Conv.html](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.GATv2Conv.html)

10. PyTorch Geometric. *PNAConv.* [https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.PNAConv.html](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.PNAConv.html)

11. Egressy, B., et al. (2023). *Provably Powerful Graph Neural Networks for Directed Multigraphs.* arXiv:2306.11586.

---

*Report generated for Graph Mining and Applications, Sapienza University of Rome, 2026.*
