# Prometheus Zero — Product Requirements Document

> **Version:** 0.1.0  
> **Date:** 2026-02-08  
> **Authors:** Gavin & Jarvis  
> **Status:** Vision & Architecture Phase

---

## Table of Contents

1. [Vision](#1-vision)
2. [Problem Statement](#2-problem-statement)
3. [Core Innovation](#3-core-innovation)
4. [Prior Art & Differentiation](#4-prior-art--differentiation)
5. [System Architecture](#5-system-architecture)
6. [Component Specifications](#6-component-specifications)
7. [Memory Architecture](#7-memory-architecture)
8. [Evaluation Architecture](#8-evaluation-architecture)
9. [Mathematical Framework](#9-mathematical-framework)
10. [Open Research Questions](#10-open-research-questions)
11. [Hardware & Infrastructure](#11-hardware--infrastructure)
12. [Roadmap](#12-roadmap)
13. [Success Criteria](#13-success-criteria)
14. [Risk Analysis](#14-risk-analysis)

---

## 1. Vision

**Create a small neural network (1–2B parameters) that discovers mathematics from scratch through self-play, using a large language model as its exploration environment — unconstrained by human mathematical traditions, notation, or cognitive biases.**

The aspiration is not to build a better theorem prover. It is to build a **mathematical explorer** — a system that, like AlphaZero in chess, might discover mathematical structures and reasoning paths that thousands of years of human mathematics never found.

### 1.1 The AlphaZero Analogy

| | AlphaGo (2016) | AlphaZero (2017) | Existing Math AI | **Prometheus Zero** |
|---|---|---|---|---|
| Training data | Human games | **None** | Human proofs/textbooks | **None** |
| Constrained by human knowledge | Yes | **No** | Yes | **No** |
| Environment | Go board | Go/Chess/Shogi board | Lean/Coq (formal) | **Large Language Model** |
| Goal | Beat humans | Beat humans | Prove human-posed theorems | **Discover new mathematics** |
| Surpassed humans | Yes | Yes (4 hours) | Partially (IMO gold) | **TBD** |

### 1.2 The Core Metaphor

> A human baby is born with zero knowledge but extraordinary learning capacity. It doesn't need knowledge injected — it needs an environment to explore and feedback to learn from.
>
> Prometheus Zero is this baby. The large language model — encoding all of human mathematical history — is its universe. Baby stands on this monument to see further, but because Baby has no preconceptions, it can look where the monument's builders never imagined.

---

## 2. Problem Statement

### 2.1 The Limitation of Current Mathematical AI

All existing mathematical AI systems learn **from** human mathematics:

- **LLMs** (GPT, Claude, DeepSeek): Trained on human text → pattern-match human reasoning → bounded by human thought patterns
- **Formal theorem provers** (AlphaProof, DeepSeek-Prover-V2): Operate within Lean/Coq → verify within human-designed axiom systems → cannot evaluate outside formal boundaries
- **Math reasoning models** (DeepSeek-V3.2-Speciale): Trained on human problems → solve human-posed questions → never generate their own questions

These systems are the mathematical equivalent of **AlphaGo** — powerful, but fundamentally limited by the human data and frameworks they absorb.

### 2.2 The Structural Constraints of Human Mathematics

Human mathematics was shaped by factors unrelated to mathematical truth:

1. **Sensory experience**: Geometry from land measurement, calculus from modeling motion
2. **Historical path dependence**: Which problems were studied first shaped which tools were developed
3. **Notation**: Arabic numerals, algebraic syntax, set-builder notation all constrain how we think
4. **Cognitive biases**: Preference for "elegance," "beauty," and human-comprehensible structures
5. **Social dynamics**: What gets funded, published, and rewarded in academia
6. **Biological limits**: Working memory constraints, sequential reasoning, finite lifespan

A system free from these constraints operates in a fundamentally larger mathematical space.

### 2.3 The Missing Piece — Now Solved

Previous attempts at "MathZero" failed because:

| Challenge | Why It Blocked Progress | Our Solution |
|-----------|----------------------|--------------|
| No rich environment for open-ended math | Formal proof assistants are rigid rule-checkers | **LLM as environment** — rich, open-ended, can evaluate novelty |
| No nuanced reward signal | Binary proof/no-proof gives sparse learning signal | **5D continuous reward** from LLM evaluation |
| No self-play mechanism for math | Math doesn't have two "players" | **Conjecturer vs. Prover** adversarial structure |
| No way to evaluate frontier discoveries | Lean goes silent at the boundary of its rules | **LLM can reason about novel mathematics** |
| Prohibitive cost of frontier model evaluation | Can't call GPT-4/Claude millions of times | **Hierarchical evaluation** with distilled reward models |

---

## 3. Core Innovation

### 3.1 Three Original Contributions

**Innovation 1: LLM as Game Board (The Monument)**

No prior work uses a large language model as the exploration environment for mathematical self-play. All existing systems use formal proof assistants (Lean, Coq, Isabelle). The LLM-as-environment paradigm enables:

- Open-ended evaluation (not limited to formal rules)
- Nuanced, multi-dimensional feedback
- Assessment of novelty, significance, and cross-domain connections
- Understanding of novel notation and abstractions that Baby may invent

**Innovation 2: Self-Directed Mathematical Discovery**

No prior work aims for autonomous mathematical exploration. All existing systems prove theorems that humans formulate. Prometheus Zero:

- Generates its own conjectures (Conjecturer)
- Attempts its own proofs (Prover)
- Builds its own theorem library (cumulative knowledge)
- Follows its own curiosity (intrinsic motivation)

**Innovation 3: Conjecturer-Prover Adversarial Self-Play**

The two-role structure creates natural adversarial dynamics:

- **Conjecturer** is rewarded for generating true, non-trivial, provable statements
- **Prover** is rewarded for finding proofs or counterexamples
- Difficulty auto-scales as both improve
- This is the mathematical analogue of AlphaZero's self-play

### 3.2 What We Are NOT Doing

To be precise about scope:

- ❌ Not building a better theorem prover (AlphaProof, DeepSeek-Prover already do this)
- ❌ Not building a better math tutor or problem solver
- ❌ Not competing on existing benchmarks (miniF2F, AIME, IMO)
- ✅ Building a system that **discovers mathematics autonomously**
- ✅ Exploring whether self-play can find mathematics **beyond human frameworks**

---

## 4. Prior Art & Differentiation

### 4.1 Existing Systems

| System | Organization | Approach | Environment | Limitation |
|--------|-------------|----------|-------------|------------|
| AlphaProof | DeepMind (2024) | RL + MCTS for Lean 4 | Lean 4 | Proves human-posed problems only |
| DeepSeek-Prover-V2 | DeepSeek (2025) | RL on cold-start data, Lean 4 | Lean 4 | Proves human-posed problems only |
| DeepSeek-V3.2-Speciale | DeepSeek (2025) | Large MoE, math-specialized | Text generation | Solves human problems, no discovery |
| HTPS | Meta (2022) | Hyper-tree proof search | Lean/Coq | Search strategy only, no self-play |
| LeanDojo | MIT (2023) | Open-source Lean interaction | Lean 4 | Infrastructure, not discovery |
| GPT-f | OpenAI (2020) | Language model for Metamath | Metamath | Early work, limited scale |

### 4.2 What We Reuse vs. What's New

| Component | Status | Source |
|-----------|--------|--------|
| RL training (GRPO) | Existing technique | DeepSeek-R1 |
| MCTS search | Existing technique | AlphaZero |
| Small language models | Existing artifacts | Qwen, DeepSeek, LLaMA |
| Math evaluation models | Existing artifacts | DeepSeek-Prover-V2, Speciale distills |
| **LLM-as-environment architecture** | **Novel** | This project |
| **Conjecturer-Prover self-play for math** | **Novel** | This project |
| **Autonomous mathematical discovery goal** | **Novel** | This project |
| **Hierarchical LLM evaluation pyramid** | **Novel** | This project |

**Analogy**: We are not inventing the engine, the wheels, or the steering system. We are designing a vehicle that has never been built — one that drives itself into territory no vehicle has ever reached.

---

## 5. System Architecture

### 5.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    PROMETHEUS ZERO SYSTEM                      │
│                                                                │
│  ┌─────────────┐     ┌─────────────┐     ┌────────────────┐  │
│  │             │     │   Theorem    │     │   Evaluation   │  │
│  │  Baby (1-2B)│◄───►│    Store     │◄───►│    Pyramid     │  │
│  │             │     │  (External   │     │                │  │
│  │ · Conjecturer│     │   Memory)   │     │ L1: Reward Model│  │
│  │ · Prover    │     │             │     │ L2: DeepSeek   │  │
│  │ · Policy    │     │ · Vector DB │     │ L3: Claude     │  │
│  │ · Value     │     │ · DAG Index │     │                │  │
│  │             │     │ · Retrieval │     │                │  │
│  └──────┬──────┘     └─────────────┘     └───────┬────────┘  │
│         │                                         │           │
│         │         ┌───────────────┐               │           │
│         └────────►│  MCTS Engine  │◄──────────────┘           │
│                   │               │                            │
│                   │  · Selection  │                            │
│                   │  · Expansion  │                            │
│                   │  · Evaluation │                            │
│                   │  · Backup     │                            │
│                   └───────┬───────┘                            │
│                           │                                    │
│                   ┌───────▼───────┐                            │
│                   │ GRPO Trainer  │                            │
│                   │               │                            │
│                   │ Policy update │                            │
│                   │ Value update  │                            │
│                   └───────────────┘                            │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow

```
1. Baby (Conjecturer) generates a conjecture φ
       │
2. Baby (Prover) attempts to prove/disprove φ using MCTS
       │
3. During MCTS, Baby queries Theorem Store for relevant prior results
       │
4. MCTS produces improved policy π and outcome z
       │
5. Evaluation Pyramid scores the attempt:
   └─ L1 (Reward Model): Basic filtering       → 90% handled here
   └─ L2 (DeepSeek): Deep math evaluation      → 9.9% handled here
   └─ L3 (Claude): Frontier research            → 0.1% handled here
       │
6. Reward R = ⟨consistency, novelty, depth, connection, elegance⟩
       │
7. If proved: theorem added to Theorem Store
       │
8. GRPO updates Baby's policy and value networks
       │
9. Repeat → Flywheel accelerates
```

---

## 6. Component Specifications

### 6.1 Baby — The Explorer (1–2B Parameters)

**Architecture**: Transformer-based with retrieval augmentation

| Component | Specification | Rationale |
|-----------|--------------|-----------|
| Base architecture | Transformer (decoder-only) | Proven for sequence reasoning |
| Parameter count | 1–2B | Sweet spot: enough for deep reasoning, small enough for fast iteration |
| Context window | 32K tokens | Long enough for multi-step proofs |
| Retrieval module | Cross-attention over external memory | Access theorem store without consuming parameters |
| Output heads | 3: Policy, Value, Conjecture | Policy guides proof steps; Value estimates success; Conjecture generates propositions |

**Base model candidates** (for initial weights before RL):

| Model | Parameters | Why Consider |
|-------|-----------|--------------|
| Qwen2.5-1.5B | 1.5B | Strong multilingual, good math base |
| DeepSeek-Prover-V2-7B | 7B | Purpose-built for math proving (larger than target, could distill) |
| Qwen3-4B-Speciale-Math-Distill | 4B | Math-distilled from Speciale |
| Train from scratch | 1-2B | True tabula rasa — no human math biases |

**Critical decision**: Starting from a pre-trained model gives faster initial progress but inherits human mathematical biases. Training from scratch is purer but much harder. **Recommendation**: Start with minimal pre-training (language understanding only, not math), then RL from scratch on mathematical exploration.

### 6.2 Theorem Store — External Memory

**Purpose**: Infinite-capacity storage for Baby's discoveries, decoupling knowledge from reasoning capacity.

```
┌─────────────────────────────────────────┐
│            THEOREM STORE                 │
│                                          │
│  ┌───────────────────────┐              │
│  │   Vector Database      │ ← Semantic search: "theorems related    │
│  │   (FAISS / Milvus)     │   to current proof state"              │
│  └───────────────────────┘              │
│                                          │
│  ┌───────────────────────┐              │
│  │   Dependency Graph     │ ← DAG: which theorems build on which   │
│  │   (NetworkX / Neo4j)   │                                        │
│  └───────────────────────┘              │
│                                          │
│  ┌───────────────────────┐              │
│  │   Utility Scores       │ ← How often each theorem is cited      │
│  │   (Citation Count)     │   in other proofs → measures value     │
│  └───────────────────────┘              │
│                                          │
│  ┌───────────────────────┐              │
│  │   Consolidation Queue  │ ← High-utility theorems → candidate   │
│  │                        │   for replay training into weights     │
│  └───────────────────────┘              │
└─────────────────────────────────────────┘
```

**Memory operations**:

| Operation | When | How |
|-----------|------|-----|
| **Write** | Baby proves a new theorem | Store theorem + proof + embedding |
| **Read** | During MCTS proof search | Query by semantic similarity to current proof state |
| **Consolidate** | Every N episodes | Replay high-utility theorems → fine-tune Baby's weights |
| **Prune** | Periodically | Remove subsumed theorems (if A implies B and B is never used alone, keep only A) |

### 6.3 MCTS Engine — The Search

Adapted from AlphaZero for proof search:

```
PUCT Selection:
  a = argmax [ Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a)) ]

Where:
  s = current proof state
  a = candidate proof step (inference rule application)
  Q(s,a) = mean value of this step from prior simulations
  P(s,a) = Baby's policy network prior
  N(s,a) = visit count
  c_puct = exploration constant (tunable)
```

**Simulations per move**: Start with 100 (vs. AlphaZero's 800 for chess). Proof search may need fewer but deeper simulations.

**Proof step actions**:

| Action Type | Description |
|-------------|-------------|
| `apply_axiom(i)` | Use axiom i |
| `apply_theorem(t)` | Use previously proven theorem t (from Theorem Store) |
| `modus_ponens(φ₁, φ₂)` | From φ₁ and φ₁→φ₂, derive φ₂ |
| `generalize(x, φ)` | Universal generalization |
| `specialize(φ, t)` | Substitute term t into universal statement |
| `induction(φ, n)` | Apply induction on n |
| `introduce_lemma(ψ)` | Propose a sub-goal (lemma) |
| `retrieve(query)` | Query Theorem Store for relevant results |

---

## 7. Memory Architecture

### 7.1 The Problem

1B parameters ≈ 2GB of weights. This must encode BOTH reasoning ability AND mathematical knowledge. As Baby discovers more mathematics, these compete for capacity.

### 7.2 The Solution — Human-Inspired Memory Hierarchy

Modeled on von Neumann architecture (which itself abstracts human memory):

| Human Memory | Computer Analogy | Baby's Implementation |
|-------------|-----------------|----------------------|
| Working memory (~7 items) | CPU registers / L1 cache | **Context window** (32K tokens) |
| Long-term memory (patterns, intuitions) | RAM | **Neural weights** (1-2B parameters) |
| External memory (books, notes) | Disk / Database | **Theorem Store** (unlimited) |

### 7.3 What Lives Where

| Content | Storage | Access Speed | Capacity |
|---------|---------|-------------|----------|
| Core reasoning strategies | Weights | Instant (forward pass) | Limited (1-2B params) |
| Frequently-used fundamental theorems | Weights (via consolidation) | Instant | Limited |
| All discovered theorems | Theorem Store (vector DB) | Fast (retrieval query) | Unlimited |
| Current proof context | Context window | Instant | 32K tokens |
| Theorem relationships | Dependency graph | Fast (graph query) | Unlimited |

### 7.4 Memory Consolidation (Sleep Replay)

Inspired by human memory consolidation during sleep:

```
Every K self-play episodes:
  1. Identify high-utility theorems (frequently cited in proofs)
  2. Construct training examples using these theorems
  3. Fine-tune Baby on these examples (consolidation)
  4. Effect: Core knowledge moves from Theorem Store → Weights
  5. Baby's "intuition" about important mathematics strengthens
```

**Key insight**: Baby doesn't need to remember everything in weights. It needs to internalize **patterns** and **core building blocks**. Everything else is in the external store.

### 7.5 Capacity Estimation

| Content | Estimated Storage Need |
|---------|----------------------|
| 10,000 theorems in vector DB | ~100MB |
| Dependency graph (10K nodes) | ~10MB |
| Full proof traces | ~1GB |
| **Total external memory** | **~1.2GB** |
| Baby's weights (1.5B, fp16) | **~3GB** |

All fits comfortably on any modern GPU alongside the model.

---

## 8. Evaluation Architecture

### 8.1 The Cost Problem

Using a frontier LLM (Claude) to evaluate every self-play episode is prohibitively expensive. At 1M episodes/day × $0.03/evaluation = $30,000/day. Unacceptable.

### 8.2 The Solution — Evaluation Pyramid

```
                 ┌───────┐
                 │Claude │  0.1% — Deep frontier research
                 │  (L3) │  ~1,000 calls/day → ~$30/day
                 └───┬───┘
                     │ Escalation: high novelty OR low L2 confidence
                ┌────┴────┐
                │DeepSeek │  9.9% — Expert math evaluation
                │  (L2)   │  Local on A100/4090 → ~$0
                └────┬────┘
                     │ Escalation: reward model uncertain
             ┌───────┴───────┐
             │ Reward Model  │  90% — Routine filtering
             │    (L1)       │  Tiny model, CPU → ~$0
             └───────────────┘
```

### 8.3 Layer Specifications

**L1 — Lightweight Reward Model (~500M parameters)**

| Attribute | Specification |
|-----------|--------------|
| Size | ~500M parameters |
| Hardware | CPU or shared GPU |
| Training data | Distilled from L2 (DeepSeek) evaluations |
| Evaluates | Well-formedness, basic validity, deduplication, trivial novelty |
| Speed | <10ms per evaluation |
| Accuracy | ~95% agreement with L2 on routine cases |

**L2 — DeepSeek Mathematical Evaluation**

| Attribute | Specification |
|-----------|--------------|
| Primary model | Qwen3-14B-DeepSeek-V3.2-Speciale-Distill (Q4 GGUF) |
| Secondary model | DeepSeek-Prover-V2-7B (for formal verification when applicable) |
| Hardware | RTX 4090 (local) or A100 time-share |
| Evaluates | Proof correctness, reasoning quality, mathematical significance, novelty assessment |
| Speed | ~1-5s per evaluation |
| Accuracy | Near-frontier for mathematical reasoning |

**L3 — Claude Deep Research**

| Attribute | Specification |
|-----------|--------------|
| Model | Claude (frontier, deep research mode) |
| Hardware | API call |
| Evaluates | True novelty assessment, cross-domain connections, significance for human mathematics, potential breakthrough identification |
| Speed | ~10-30s per evaluation |
| Triggers | (1) L2 confidence < threshold, (2) novelty score > threshold, (3) periodic calibration |

### 8.4 Escalation Logic

```python
def evaluate(baby_output):
    # L1: Fast filter
    l1 = reward_model.evaluate(baby_output)
    
    if l1.is_malformed or l1.is_duplicate:
        return Reward(0)  # Reject immediately
    
    if l1.confidence > 0.95 and l1.novelty_estimate < 0.3:
        return l1.to_reward()  # ~90% of cases end here
    
    # L2: Deep math evaluation
    l2 = deepseek.evaluate(baby_output)
    
    if l2.confidence > 0.9 and l2.novelty_estimate < 0.7:
        return l2.to_reward()  # ~9.9% of cases end here
    
    # L3: Frontier research — only the most promising 0.1%
    l3 = claude.deep_research(
        baby_output,
        context=theorem_store.get_related(baby_output),
        question="Is this a genuine mathematical discovery? "
                 "What connections exist to known mathematics? "
                 "Is this worth pursuing further?"
    )
    
    return l3.to_reward()
```

### 8.5 Reward Model Training Loop

```
Phase 1: Bootstrap
  - Run Baby for N episodes with L2 (DeepSeek) evaluating everything
  - Collect (input, L2_score) pairs
  - Train L1 reward model on this data

Phase 2: Calibration
  - L1 handles 90% of traffic
  - L2 handles escalations + periodic random samples
  - Compare L1 vs L2 on random samples → fine-tune L1 if drift detected

Phase 3: Frontier Calibration  
  - L3 (Claude) periodically reviews L2's highest-novelty evaluations
  - Ensures the entire pyramid stays calibrated as Baby's explorations become more sophisticated
```

### 8.6 Cost Estimate

| Layer | Daily Volume (1M episodes) | Unit Cost | Daily Cost |
|-------|---------------------------|-----------|------------|
| L1 Reward Model | 900,000 | $0 (local CPU) | **$0** |
| L2 DeepSeek | 99,000 | $0 (local GPU) | **~$0** (electricity) |
| L3 Claude | 1,000 | ~$0.03 | **~$30** |
| **Total** | | | **~$30/day** |

---

## 9. Mathematical Framework

### 9.1 State Space

```
s_t = (T_t, φ_t, Π_t)

T_t = theorem library at time t (Baby's cumulative discoveries)
φ_t = current proposition under investigation  
Π_t = current partial proof state
```

### 9.2 Neural Network

```
f_θ(s) → (p, v, φ)

p ∈ R^|A|  — policy: probability over proof actions
v ∈ [-1,1] — value: estimated probability of proof success
φ           — conjecture: proposed new proposition
```

### 9.3 MCTS Selection (PUCT)

```
a_t = argmax_a [ Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a)) ]
```

### 9.4 Self-Play Reward Structure

**Conjecturer-Prover dynamics:**

| Outcome | Conjecturer Reward | Prover Reward |
|---------|-------------------|---------------|
| True, proved | +r (good problem) | +r (solved) |
| True, unproved | +ε (too hard) | -r (failed) |
| False, disproved | -r (bad conjecture) | +r (found counterexample) |
| Trivially true | -r (boring) | 0 |

**5D evaluation reward from the Pyramid:**

```
R(φ, Π) = w₁·r_consistency + w₂·r_novelty + w₃·r_depth + w₄·r_connection + w₅·r_elegance
```

| Dimension | Definition | Measures |
|-----------|-----------|----------|
| r_consistency | Does not contradict known math | Correctness |
| r_novelty | Not implied by existing theorems in T_t | Genuine discovery |
| r_depth | Distance from axioms in proof graph | Depth of exploration |
| r_connection | Relates to known open problems or active areas | Significance |
| r_elegance | Shortens proofs of other theorems | Structural insight |

**Elegance reward (compression bonus):**
```
r_elegance(φ) = Σ_{ψ∈T} [ proof_len(ψ | T\{φ}) - proof_len(ψ | T) ]
```

### 9.5 Training Loss

```
L(θ) = (z - v)² - π^T · log(p) + c·‖θ‖²

z = actual outcome (+1 proved, -1 disproved, 0 failed)
v = predicted value
π = MCTS-improved policy
p = network raw policy
c = regularization coefficient
```

### 9.6 The Flywheel

```
Better f_θ → Better MCTS → Better self-play data
  → Better training → Better f_θ
  → Deeper theorems → Richer T_t → More composable tools
  → Exponentially larger explorable space → ...
```

**Critical difference from game-playing**: Each discovery is permanent. T_t only grows. The flywheel accelerates.

---

## 10. Open Research Questions

### 10.1 Critical Questions (Must Resolve Before Phase 1)

| # | Question | Why It Matters | Proposed Approach |
|---|----------|---------------|-------------------|
| Q1 | What formal/informal language does Baby use to express mathematics? | Determines the entire action space | Start with lightweight formal language; allow Baby to extend it |
| Q2 | How to define the axiom set Baby starts from? | Too few = can't derive anything; too many = constraining | Start minimal (Peano arithmetic), expand if needed |
| Q3 | How to prevent Baby from staying in trivial territory? | Could prove x=x a million times | Novelty reward + curiosity bonus + diminishing returns on similar proofs |
| Q4 | What's the right balance between Conjecturer and Prover training? | One could dominate the other | Alternate training; monitor both reward curves |

### 10.2 Important Questions (Can Iterate During Development)

| # | Question | Proposed Approach |
|---|----------|-------------------|
| Q5 | Optimal Baby size (1B vs 2B vs 4B)? | Start with 1.5B, scale up if reasoning depth is insufficient |
| Q6 | How many MCTS simulations per proof attempt? | Start with 100, tune based on proof success rate |
| Q7 | How to handle Baby inventing new notation? | Let it; train L1/L2 evaluators to parse Baby's output format |
| Q8 | When should consolidation (sleep replay) happen? | Every 10K episodes; tune based on performance curves |
| Q9 | Can Baby discover mathematics that contradicts its starting axioms? | Allow Baby to propose alternative axioms; evaluate with L3 |

### 10.3 Philosophical Questions (Long-term)

| # | Question |
|---|----------|
| Q10 | If Baby discovers novel mathematics, how do we verify it's meaningful and not just syntactically valid nonsense? |
| Q11 | Could Baby empirically discover the boundaries of Gödel's incompleteness? |
| Q12 | Does self-play mathematical discovery constitute a form of mathematical understanding? |
| Q13 | If Baby develops its own notation, is it "doing mathematics" or something else entirely? |

---

## 11. Hardware & Infrastructure

### 11.1 Available Hardware

| Machine | GPU | RAM | Role |
|---------|-----|-----|------|
| A100 Server | A100-SXM4-80GB | 251GB | Baby training (primary) |
| GPU Server #2 | RTX 4090 24GB × 2 | — | L2 inference (DeepSeek + Prover-V2) |

### 11.2 Resource Allocation

**Training Phase:**
```
A100 80GB:
├── Baby (1.5B) training     ~30GB (model + optimizer + gradients)
├── MCTS rollout buffer       ~5GB
├── Theorem Store (in-GPU)    ~2GB
└── Headroom                  ~43GB

4090 #1:
├── Qwen3-14B-Speciale-Distill (Q4)  ~8GB
└── Headroom for KV cache             ~16GB

4090 #2:
├── DeepSeek-Prover-V2-7B (fp16)     ~14GB
└── Headroom for KV cache             ~10GB
```

**Cost:**
```
Electricity: ~$5/day
Claude API (L3): ~$30/day
Total: ~$35/day
```

### 11.3 Software Stack

| Component | Technology |
|-----------|-----------|
| Training framework | PyTorch + custom GRPO implementation |
| MCTS engine | Custom (Python + C++ for speed-critical paths) |
| Vector database | FAISS (in-process, GPU-accelerated) |
| Graph database | NetworkX (in-memory) or SQLite for persistence |
| Model serving (L2) | vLLM or llama.cpp |
| Experiment tracking | Weights & Biases |
| Version control | Git (theorem store snapshots) |

---

## 12. Roadmap

### Phase 0: Foundation (Weeks 1–2)

| Task | Description | Deliverable |
|------|------------|-------------|
| P0.1 | Define Baby's mathematical language (action space) | Language specification doc |
| P0.2 | Implement Theorem Store (vector DB + graph) | Working storage system |
| P0.3 | Implement MCTS engine for proof search | Working MCTS with toy environment |
| P0.4 | Set up GRPO training pipeline | Training loop on synthetic task |
| P0.5 | Deploy L2 models (DeepSeek + Prover-V2) on 4090s | Running inference endpoints |
| P0.6 | Build evaluation pipeline (L1 → L2 → L3) | End-to-end evaluation flow |

### Phase 1: Arithmetic Playground (Weeks 3–4)

| Task | Description | Success Criterion |
|------|------------|-------------------|
| P1.1 | Initialize Baby from minimal base | Model loads and generates |
| P1.2 | Define Peano axioms as starting state | Axioms formalized in Baby's language |
| P1.3 | Run self-play on natural number arithmetic | Baby rediscovers commutativity, associativity |
| P1.4 | Validate flywheel: training improves proof success rate | Monotonically improving success curve |
| P1.5 | Validate Theorem Store grows meaningfully | >100 non-trivial theorems discovered |

### Phase 2: Self-Curriculum (Weeks 5–8)

| Task | Description | Success Criterion |
|------|------------|-------------------|
| P2.1 | Activate Conjecturer-Prover adversarial training | Both roles improve over time |
| P2.2 | Implement curiosity reward | Baby explores beyond arithmetic |
| P2.3 | Implement elegance reward (compression bonus) | Baby discovers simplifying lemmas |
| P2.4 | Implement memory consolidation (sleep replay) | Core theorems persist in weights |
| P2.5 | Scale to 1M episodes/day | Infrastructure handles throughput |

### Phase 3: Open Exploration (Weeks 9–16)

| Task | Description | Success Criterion |
|------|------------|-------------------|
| P3.1 | Remove domain constraints | Baby explores freely |
| P3.2 | Monitor for novel discoveries | L3 identifies at least one genuinely novel result |
| P3.3 | Analyze Baby's theorem library vs. human mathematics | Comparison report |
| P3.4 | Extend axiom set if Baby hits ceiling | Richer starting point, new exploration |

### Phase 4: Beyond (Week 17+)

| Task | Description |
|------|------------|
| P4.1 | Deep analysis of Baby's discovered mathematical structures |
| P4.2 | Human mathematician review of most significant findings |
| P4.3 | Paper: "Mathematical Discovery Through Self-Play with LLM Environments" |
| P4.4 | Open-source release of framework + discovered theorems |

---

## 13. Success Criteria

### 13.1 Minimum Viable Success

- [ ] Baby independently rediscovers basic arithmetic properties from Peano axioms
- [ ] Theorem Store contains >1,000 non-trivial, Baby-discovered theorems
- [ ] Flywheel effect demonstrated: later training epochs produce deeper theorems than earlier ones
- [ ] Cost remains under $50/day

### 13.2 Strong Success

- [ ] Baby discovers at least one theorem or proof path that is not in standard textbooks
- [ ] Baby's proof style is measurably different from human proofs (shorter, different structure)
- [ ] Evaluation pyramid achieves >95% agreement between L1 and L2 on routine cases
- [ ] Framework is generalizable to other domains

### 13.3 Breakthrough Success

- [ ] Baby discovers a genuinely novel mathematical structure or relationship
- [ ] Discovery is verified by human mathematicians as meaningful
- [ ] Baby's exploration path diverges significantly from human mathematical history
- [ ] Paper accepted at top-tier venue (NeurIPS, ICML, Nature)

---

## 14. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Baby stays in trivial territory (proves x=x forever) | Medium | High | Novelty reward + curiosity bonus + diminishing returns |
| LLM evaluator hallucinates (approves incorrect math) | Medium | High | Cross-check with formal tools where possible; L3 calibration |
| Flywheel doesn't engage (no improvement over time) | Low-Medium | Critical | Careful reward shaping; ablation studies; early detection metrics |
| Baby converges to human mathematics (no novelty) | Medium | Medium | Not actually a failure — validates the approach; push for longer training |
| Conjecturer generates only trivial or false statements | Medium | High | Careful reward balance; ensure Conjecturer is penalized for trivial/false |
| Cost exceeds budget | Low | Medium | Aggressive L1 filtering; reduce L3 calls; batch evaluations |
| 1B parameters insufficient for deep reasoning | Medium | Medium | Scale to 2-4B; add external memory; longer MCTS search |

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| Baby | The small (1-2B) neural network that explores mathematics |
| Monument | The large language model serving as Baby's exploration environment |
| Theorem Store | External memory system storing Baby's discovered theorems |
| Conjecturer | Baby's role when generating new mathematical propositions |
| Prover | Baby's role when attempting to prove/disprove propositions |
| Evaluation Pyramid | Three-tier evaluation system (L1 Reward Model → L2 DeepSeek → L3 Claude) |
| Flywheel | Self-reinforcing cycle: better model → better search → better data → better model |
| Consolidation | Process of replaying important discoveries to strengthen Baby's weights |
| GRPO | Group Relative Policy Optimization — RL algorithm for training |
| PUCT | Predictor + Upper Confidence bounds applied to Trees — MCTS selection formula |

## Appendix B: References

1. Silver, D. et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (AlphaZero)
2. Silver, D. et al. (2017). "Mastering the Game of Go without Human Knowledge" (AlphaGo Zero)
3. Xin, H. et al. (2025). "DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition"
4. Shao, Z. et al. (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
5. Trinh, T. et al. (2024). "Solving Olympiad Geometry without Human Demonstrations" (AlphaGeometry)
6. DeepMind (2024). "AlphaProof: AI Achieves Silver-Medal Standard Solving International Mathematical Olympiad Problems"

---

*"The only way to discover the limits of the possible is to go beyond them into the impossible." — Arthur C. Clarke*
