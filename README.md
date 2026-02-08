# Prometheus Zero

**A self-playing mathematical mind that discovers mathematics from scratch, using large language models as its universe.**

> *Prometheus stole fire from the gods and gave it to humanity.  
> We steal mathematical truth from the universe — and give it back.*

---

## Vision

AlphaZero mastered chess in 4 hours without studying a single human game. It discovered strategies that 1,500 years of human theory never found — because it was **unconstrained by human habits of thought**.

We ask: **Can the same be done for mathematics?**

Prometheus Zero is a small (1–2B parameter) neural network that:

- Starts with **zero mathematical knowledge**
- Explores mathematical space through **self-play**
- Uses a **large language model as its environment** — the "board" on which it plays
- Discovers theorems, proofs, and potentially entire mathematical structures that humans have never conceived

The large model is not the student. **It is the universe in which the student explores.**

---

## The Problem

All existing mathematical AI systems share a fundamental limitation: **they learn from human mathematics.**

- LLMs learn from textbooks and papers → bounded by human thought patterns
- Formal theorem provers (Lean, Coq) verify within human-designed axiom systems
- Neural theorem provers train on human proofs → inherit human biases

Human mathematics was shaped by:

- **Sensory experience** — geometry from land measurement, calculus from motion
- **Historical accidents** — which problems were considered important
- **Notation** — Arabic numerals and algebraic notation shape how we think
- **Cognitive biases** — we favor theories that feel "elegant"
- **Social factors** — what gets funded, what gets published

A system free from these constraints might discover mathematics that is fundamentally different — not wrong, but *different*. And in mathematics, "different" means **"new"**.

---

## The Key Insight

Previous attempts to build a "MathZero" failed because:

1. **The environment was too rigid** — formal proof assistants can only verify within human-defined rules. When the explorer reaches the frontier, the verifier goes silent.
2. **The reward was too sparse** — binary proof completion (proved/not proved) provides almost no learning signal.
3. **No good self-play mechanism** existed for open-ended mathematics.

**Our insight: A large language model can serve as all three — environment, reward, and opponent.**

A frontier LLM:

- **IS** a rich environment — it encodes the full landscape of human mathematical knowledge
- **CAN** provide nuanced rewards — not just right/wrong, but "interesting", "novel", "this connects to Ramanujan's work"
- **CAN** evaluate at the frontier — where formal systems have no rules, an LLM can still reason
- **CAN** understand novel notation — if Baby invents its own abstractions, a language model can attempt to interpret them

The formal proof assistant is a law book. The large model is a wise teacher. **We need the teacher.**

---

## Architecture

```
                    ┌─────────────────────┐
                    │     Baby (1–2B)     │
                    │    The Explorer     │
                    │                     │
                    │  · Policy head      │
                    │  · Value head       │
                    │  · Conjecture head  │
                    └─────────┬───────────┘
                              │
                     generates conjectures
                      attempts proofs
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Large Model (LLM)  │
                    │   The Monument      │
                    │                     │
                    │  Evaluates:         │
                    │  · Consistency      │
                    │  · Novelty          │
                    │  · Depth            │
                    │  · Connections      │
                    │  · Elegance         │
                    │                     │
                    │  Returns: 5D reward │
                    └─────────┬───────────┘
                              │
                     rich feedback signal
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Baby updates      │
                    │   policy via GRPO   │
                    │   + MCTS for search │
                    └─────────────────────┘
```

### The Monument — LLM as Game Board

Why not a formal proof assistant?

| Capability | Lean 4 / Coq | Large Language Model |
|-----------|--------------|---------------------|
| Verify known-rule proofs | ✅ | ✅ |
| Evaluate novelty | ❌ | ✅ |
| Assess connections to open problems | ❌ | ✅ |
| Handle novel notation and abstractions | ❌ | ✅ |
| Provide directional feedback | ❌ | ✅ |
| Evaluate beyond formal boundaries | ❌ | ✅ |

Lean is a law book — it tells you what's legal within its rules. The Monument is a universe — it tells you what's interesting, what's new, and where to look next.

### Self-Play — Conjecturer vs. Prover

Mathematics doesn't have two "players" like chess. We create adversarial dynamics through **role-splitting**:

| Outcome | Conjecturer Reward | Prover Reward |
|---------|-------------------|---------------|
| True, and proved | +r (good problem) | +r (solved it) |
| True, but unproved | +ε (too hard) | −r (failed) |
| False, disproved | −r (bad conjecture) | +r (found counterexample) |
| Trivially true | −r (boring) | 0 |

This mirrors AlphaZero's self-balancing dynamics:

- The **Conjecturer** learns to generate statements that are true, interesting, and at the edge of the Prover's ability
- The **Prover** learns increasingly sophisticated reasoning strategies
- Difficulty **auto-scales** with capability — just like always playing an equal opponent

---

## Mathematical Framework

### State Space

```
s_t = (T_t, φ_t, Π_t)
```

- **T_t** — Theorem library at time *t* (discovered by Baby, not imported from humans)
- **φ_t** — Current proposition under investigation
- **Π_t** — Current partial proof state

### Neural Network

```
f_θ(s) → (p, v, φ)
```

- **p** — Policy: probability distribution over reasoning actions
- **v** — Value: estimated probability of proving the current proposition
- **φ** — Conjecture: proposed new proposition to investigate

### Monte Carlo Tree Search for Proofs

```
a_t = argmax_a [ Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a)) ]
```

Identical to AlphaZero's PUCT formula, applied to proof steps instead of board moves:

- **Q(s,a)** — Historical value of this reasoning step (exploitation)
- **P(s,a)** — Network's intuition about this step (guidance)
- **√N(s) / (1 + N(s,a))** — Exploration bonus for untried paths

### Reward Function

The Monument provides a **5-dimensional continuous reward** — not binary verification:

```
R(φ, Π) = ⟨ r_consistency, r_novelty, r_depth, r_connection, r_elegance ⟩
```

| Dimension | What It Measures | Why It Matters |
|-----------|-----------------|----------------|
| Consistency | Does this contradict known mathematics? | Filters nonsense |
| Novelty | Is this already known? Trivially derivable? | Rewards genuine discovery |
| Depth | How far from axioms? How deep is the reasoning? | Rewards reaching further |
| Connection | Does this relate to known open problems? | Rewards significance |
| Elegance | Does this compress or simplify existing knowledge? | Rewards structural insight |

**Elegance reward, formally:**

```
r_elegance(φ) = Σ_ψ∈T [ proof_len(ψ | T\{φ}) − proof_len(ψ | T) ]
```

A theorem that **shortens the proofs of other theorems** is rewarded. This is a mathematical formalization of elegance: good mathematics makes the entire system simpler.

### Training

```
L(θ) = (z − v)² − π^T · log(p) + c·‖θ‖²
```

- **(z − v)²** — Value loss: learn to predict proof outcomes
- **−π^T · log(p)** — Policy loss: learn to match MCTS-improved policy
- **c·‖θ‖²** — L2 regularization

### The Flywheel

```
Better network
  → Better MCTS search
  → Stronger self-play
  → Higher quality conjectures and proofs
  → Better training data
  → Better network
  → Deeper theorems discovered
  → Richer theorem library (permanent tools!)
  → Exponentially larger explorable space
  → ...
```

**Critical difference from chess**: In chess, each game starts from an empty board. In mathematics, each discovery is a **permanent tool** for future discoveries. Knowledge compounds. The flywheel accelerates over time.

---

## Why the Monument, Not a Textbook

The large model stands on the shoulders of all human mathematical history. Every theorem, every proof technique, every heuristic, every failed conjecture — compressed into its weights.

Baby stands on the Monument to see further. But because Baby has no preconceptions — no habits, no aesthetic preferences, no path dependence — it can look in directions the Monument's builders never imagined.

```
         Baby's gaze →  ? ? ? (unexplored mathematical space)
              ↑
        ┌───────────┐
        │   Baby    │  (unconstrained, free)
        └─────┬─────┘
              │  stands on
        ┌─────┴─────┐
        │ Monument  │  (all human mathematical knowledge)
        │   (LLM)   │
        └─────┬─────┘
              │  built from
    ┌─────────┴─────────┐
    │  Human Mathematics │
    │  (3,000+ years)    │
    └───────────────────┘
```

**Newton said: "If I have seen further, it is by standing on the shoulders of giants."**

Prometheus Zero stands on a giant that has already absorbed all previous giants — and looks where none of them ever looked.

---

## What Baby Might Discover

### Likely

- Fundamental arithmetic properties (commutativity, associativity) — rediscovered independently, possibly via entirely different reasoning paths
- Shorter proofs of known theorems (human proofs carry historical baggage)

### Possible

- Powerful intermediate lemmas that humans overlooked (too "ugly" to publish, but immensely useful as tools)
- Novel connections between mathematical areas that humans treat as separate
- Alternative foundational preferences — different axiom sets that generate the same (or different) mathematics

### Speculative but Non-Zero

- Empirical intuitions about Gödelian boundaries — which statement structures tend to be undecidable
- Mathematical structures with no human name, no human intuition, no human analogue
- Entirely new branches of mathematics that require no human concepts to formulate

---

## Scalable Evaluation

Using a frontier LLM as the environment for millions of evaluations is expensive. The solution is **hierarchical evaluation**:

```
Layer 1 (99% of traffic):  Lightweight reward model
                           (distilled from the Monument's evaluations)

Layer 2 (~1%):             Full Monument (frontier LLM)
                           Deep evaluation of promising discoveries

Layer 3 (rare):            Human mathematicians
                           Review genuinely novel findings
```

This mirrors the structure of human academia:

- Most papers are reviewed by peers (reward model)
- Breakthrough candidates reach top experts (Monument)
- Revolutionary results are examined by the global community (humans)

The reward model is trained on thousands of the Monument's deep evaluations, learning to approximate its 5D judgment. The Monument periodically recalibrates the reward model as Baby's explorations become more sophisticated.

---

## Roadmap

### Phase 0: Foundation
- [ ] Select base architecture (1–2B transformer with external memory)
- [ ] Design the LLM evaluation protocol
- [ ] Build self-play infrastructure
- [ ] Implement GRPO training pipeline

### Phase 1: Arithmetic Playground
- [ ] Baby explores natural number arithmetic from Peano axioms
- [ ] Rediscover basic properties (commutativity, associativity, distributivity)
- [ ] Validate the self-play training loop

### Phase 2: Self-Curriculum
- [ ] Conjecturer / Prover adversarial training
- [ ] Curiosity-driven exploration (novelty bonus)
- [ ] Automatic difficulty scaling

### Phase 3: Open Exploration
- [ ] Remove domain constraints
- [ ] Let Baby explore freely across mathematical space
- [ ] Monitor for novel discoveries

### Phase 4: Beyond
- [ ] Analyze Baby's discovered mathematics
- [ ] Compare development path with human mathematical history
- [ ] Publish findings

---

## Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Training GPU | A100 40GB | A100 80GB |
| Inference GPU | RTX 3090 24GB | RTX 4090 24GB |
| LLM Environment | API access to frontier model | Local 70B+ or frontier API |
| RAM | 64GB | 128GB+ |
| Storage | 500GB SSD | 1TB NVMe |

---

## The Name

**Prometheus** stole fire from the gods and gave it to humanity, fundamentally transforming civilization.

**Zero** signifies:
1. Starting from zero knowledge — *tabula rasa*
2. Lineage from AlphaZero's paradigm of self-play discovery

Together: **stealing mathematical fire from the universe, starting from nothing.**

---

## Contributing

This project is in its earliest conceptual stage. We welcome:

- Theoretical analysis of the framework
- Ideas for the reward function design
- Experience with RL training for language models
- Perspectives from mathematics, philosophy of mind, and AI safety

Open an issue or start a discussion. Every insight helps.

---

## License

MIT

---

*"The important thing is not to stop questioning. Curiosity has its own reason for existing."*  
*— Albert Einstein*

**Prometheus Zero: Mathematics, rediscovered.**
