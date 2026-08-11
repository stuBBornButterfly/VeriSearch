# VeriSearch — Kickoff

**From:** Ayesha Siddika (project lead)
**To:** Tahsinul Haque Dhrubo
**Read this first. `PROJECT.md` is the full contract; this is why it exists.**

---

## The problem

Ask an AI a hard question: *"Which director of the 1999 film that won Best Picture was born in Australia?"*

The AI cannot answer from memory. It has to look things up. Two things go wrong today.

*It searches badly.* Normal RAG searches once, using the original question, then answers. This question needs two searches — first find the film, then find the director's birthplace. One search is not enough.

*It makes things up.* When search results are weak, the model fills the gap with a confident guess. It says something that sounds right and is in none of the documents it read.

## What we are building

A model that learns to search the way a person researches something.

Think, search, read, think again, search again, answer when it actually has enough. Not a fixed script — a learned habit.

```
Question → think → search → read → think → search → read → answer
```

The model decides how many loops. Easy question, one search. Hard question, three. **That decision is the thing being learned.**

## How it learns

The reinforcement learning part, and it is simpler than it sounds.

Let the model try the same question 8 different ways. Some attempts search once and get it wrong. Some search four times, get it right, waste effort. Some search twice and nail it.

Score every attempt. Nudge the model toward the patterns that scored well, away from the ones that scored badly. Repeat thousands of times. That is GRPO.

Nobody tells it "search twice here." It finds its own strategy from the scores.

## The scoring, and why it is the interesting part

Most work scores only *was the answer right*. We score three things.

*Right answer.* Compare to the known correct answer. Straightforward.

*Honest answer.* Take each claim in the answer and check whether the retrieved documents actually support it. Claims with no supporting document get punished. This is the anti-hallucination lever, and most projects skip it.

*Cheap answer.* Small penalty per search. Otherwise the model learns to search ten times for everything — accurate and useless in production.

Three goals pulling against each other. The model has to find a balance. **Finding that balance is the research.**

## The experiment

The search penalty has a dial (`λ`). We train four models at four settings, plot accuracy against searches per question, and see what shape the curve has.

We do not know the shape yet. The obvious guess is a smooth trade-off — turn the dial up, fewer searches, lower accuracy. It might also be flat over a wide range, meaning most searches buy nothing. A flat curve is a *better* result than the obvious one, and it is a real finding either way.

That curve is the deliverable. It answers "how much accuracy does each search actually buy," which every company running RAG wants to know and nobody has measured properly.

**Rule that follows from this:** we never write the shape of the curve before the runs finish. No number in any doc, slide, or CV bullet that did not come out of a logged run.

## Why it is worth six weeks

Two things it proves that a CV cannot currently prove.

You can train a model with RL, not just call one through an API. That is the gap.

And we treat hallucination as something to measure and reduce with a number, not hand-wave about in a README. That is what separates an AI engineer from someone who wired up LangChain.

---

## Who does what

**I build the library.** Documents and corpus, the search engine and index, the scoring tools, the evaluation harness and baselines, serving, CI, the final report. Roughly 70%.

**You build the brain.** How the model takes its turns, how attempts get scored, and the training loop that makes it improve. Specifically `verisearch/env/`, `verisearch/rewards/`, `verisearch/train/`, plus `writeup/rl_method.md` and `writeup/failure_cases.md`. Roughly 30%.

Smaller share of the work, larger share of the interesting part — and the load-bearing part. Nothing downstream exists until your training loop runs. That is the trade.

Anything outside those directories: open a card, do not open an editor. If you need my code before it exists, stub it against the protocol and tell me.

---

## What you owe, and when

Each phase ends when something *runs*, not when the code looks finished. No phase starts before the previous one passes.

**P1 — Contracts and environment.**
We agree `verisearch/types.py` together in the first session. Then you build the output parser and the episode loop. Done when: the untrained base model runs 100 questions against a fake search engine and produces structured trajectory records I can open and read.

**P2 — Reward.**
All reward components, composed. Done when: the reward spread over base-model rollouts is non-degenerate (not all zeros, not all ones — roughly 15–60% scoring non-trivially), and we have both hand-read 30 scored trajectories and agree the scores match our judgment. If we disagree on a score, the reward is wrong, not the judgment.

**P3 — Training.**
The GRPO loop, loss masking, overfit test. Done when: the model overfits 20 questions (proves the loop learns at all), then a full 1.5B run beats the base model on held-out F1.

**P4 — Sweep.**
Four training runs at four `λ` values on 3B. Done when: all four finish and the W&B runs are linked. I evaluate and plot.

**P6 — Writeup.**
`rl_method.md` and `failure_cases.md`.

---

## Three things that will bite you

**1. Loss masking. Read this twice.**

Retrieved passages get pasted into the model's context, but the model did not write them. If those tokens carry loss, the model learns to predict its own search results — it learns to hallucinate retrieval. The training curves look perfect. Everything downstream is garbage.

`tests/test_masking.py` is the guard. It is blocking in CI. **Write it before the training loop, not after.**

**2. Faithfulness scoring is mine, and it runs offline.**

I give you a `FaithfulnessScorer` behind a protocol. You call it. Do not build an LLM judge inside the training loop — it is slow and it is not verifiable. Scoring happens against the passages that were actually retrieved, after the fact.

**3. The two-day timebox on the RL library.**

Multi-turn RL is not the default path in most RL libraries. Standard GRPO assumes single-turn: prompt in, completion out. Ours generates, pauses for retrieval, continues, all as one trajectory.

Three routes: TRL with a custom rollout hook, a purpose-built agentic-RL library, or hand-rolling GRPO in ~300 lines. You get two days to evaluate all three against the versions actually installed. On day three you hand-roll. That is decided now so it is not a negotiation later.

---

## Reward hacking is a deliverable, not a failure

The model will find ways to game the reward. It will stop searching and answer from memory. It will copy passages verbatim to farm the faithfulness score. It will search once and ignore the results.

When that happens: **log it, do not quietly patch it.** Write it up in `failure_cases.md` the same day — you will forget the details by the next one. Documented hacks with mitigations are the most valuable content in the report. A silently patched hack is worth nothing to anyone.

The metrics that catch these are in `PROJECT.md` §6. Watch average searches per rollout, answer length, and whether faithfulness goes flat.

---

## How we work

- **Cards before code.** Nothing starts without a Trello card in `Ready` assigned to you.
- **One card, one branch, one PR.** Keep PRs under ~400 changed lines. Small PRs get reviewed same day; large ones sit.
- **You open PRs, I review and merge.** Never self-merge. `main` is protected for both of us.
- **Green before review.** A red Jenkins pipeline is not reviewable. Jenkins is mine — you never modify `Jenkinsfile`.
- **Blocked more than 24 hours** → move the card to `Blocked`, comment why, tag me. Silence is the only unrecoverable failure mode here.
- **Weekly: one thing I can open.** Not a status update. A merged PR, a file of trajectories, a failing test that proves a real problem. Async by default, one 30-minute sync a week.
- **Read raw trajectories after every significant change.** Metrics hide behavior. The first reward hack will be obvious in ten rollouts and invisible in the mean.
- **Small first.** 1.5B before 3B. 20 questions before 2000. 100 eval examples before 500.

---

## First session — 45 minutes

1. We read this document together, out loud. Five minutes.
2. You restate the reward design back to me in your own words. If it does not survive that, we fix it now, not in P2.
3. **We agree `types.py` together.** This is the real work of the meeting — 20 minutes. `Passage`, `Trajectory`, `Turn`, `FaithfulnessResult`, and the three protocols (`Retriever`, `FaithfulnessScorer`, `RewardComponent`). We both code against these from hour one. Frozen at the end of P1; changes after that need a PR we both approve.
4. Fakes, both directions. You build a fake `Retriever` immediately and never wait on my index. I build fake trajectories and never wait on your loop. Neither of us is ever blocked on the other's implementation.
5. Two-day timebox starts.
6. I create the P1 cards.

Questions before we start are cheap. Questions after a week of wrong work are not. Ask.
