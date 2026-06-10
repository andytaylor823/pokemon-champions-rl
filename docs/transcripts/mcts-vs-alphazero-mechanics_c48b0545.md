# MCTS vs AlphaZero Mechanics

**Source:** `agent-transcripts/c48b0545-8a08-4ea5-ab5d-8e5eafb02452/c48b0545-8a08-4ea5-ab5d-8e5eafb02452.jsonl`

---

## User (Turn 1)

I want to make a reinforcement learning system / agent / model / whatever it's called to play competitive Pokemon battles, VGC style (bring 6, choose 4, double battle). But first, I want to better understand how modern RL is done. Are people still using Deep Q Networks? What are some modern RL algorithms? Explain some to me

---

## Assistant (Turn 1)

Great question — let me give you a solid overview of where RL stands today, especially as it relates to building something like a Pokemon VGC agent.

## Are People Still Using DQN?

Short answer: **rarely in serious projects anymore**. DQN (2013-2015 era) was groundbreaking, but it has well-known limitations:

- It only works with **discrete action spaces** (fine for Pokemon, but still)
- It struggles with **high-dimensional or combinatorial action spaces** (choosing 4 from 6, then picking moves for 2 Pokemon simultaneously = large action space)
- It's **sample-inefficient** — needs millions of interactions to learn
- The "max" operator in Q-learning causes **overestimation bias**

DQN variants like **Double DQN**, **Dueling DQN**, and **Rainbow** improved on it, but the field has largely moved past value-only methods for complex problems.

---

## Modern RL Algorithms (Most Relevant to You)

### 1. PPO (Proximal Policy Optimization) — The Workhorse

PPO (2017, OpenAI) is arguably the **most widely used RL algorithm today**. It's what trained OpenAI Five (Dota 2), ChatGPT (via RLHF), and countless game agents.

**How it works:**
- It's a **policy gradient** method — instead of learning "how good is this state?" (like DQN), it directly learns "what action should I take?"
- It uses an **actor-critic** architecture: one network (actor) picks actions, another (critic) estimates how good the current state is
- The key innovation is a **clipped objective function** that prevents the policy from changing too drastically in one update. This makes training stable without being overly conservative

**Why it's popular:**
- Extremely stable training (rarely diverges)
- Works with discrete and continuous action spaces
- Simple to implement and tune
- Scales well with parallelism (run many environments simultaneously)

**For Pokemon:** PPO would be a very solid choice. You'd model the policy as a neural network that takes the battle state as input and outputs a probability distribution over legal actions.

---

### 2. IMPALA / APPO (Distributed Variants)

When you need to scale PPO across many parallel environments (which you'd want for Pokemon — self-play requires tons of games), **IMPALA** (Importance Weighted Actor-Learner Architecture) and **Asynchronous PPO** decouple data collection from learning. Workers play games, a central learner trains on the data.

---

### 3. SAC (Soft Actor-Critic)

SAC (2018) is the go-to for **continuous action spaces** (robotics, etc.). It adds an **entropy bonus** to encourage exploration — the agent tries to be as random as possible while still maximizing reward. Less relevant for Pokemon (discrete actions), but worth knowing about.

---

### 4. MuZero / AlphaZero — Planning + Learning

This is the DeepMind family that mastered Chess, Go, Shogi, and Atari.

**How it works:**
- Combines a **learned model** of the environment with **Monte Carlo Tree Search (MCTS)**
- AlphaZero: knows the game rules, uses MCTS to plan ahead
- MuZero: **learns** the game dynamics without being told the rules
- Uses self-play to generate training data

**Key idea:** Instead of just reacting (like PPO), these agents **plan ahead** by simulating future states in their learned model, then pick the best path.

**For Pokemon:** This is conceptually appealing because Pokemon is a **game of imperfect information** (you don't know opponent's moves, items, EVs). However:
- MCTS assumes you can simulate forward, which is hard with hidden information
- The branching factor in VGC is enormous (2 Pokemon each choosing moves simultaneously)
- Much more complex to implement than PPO

---

### 5. Dreamer (v1, v2, v3) — World Models

**How it works:**
- Learns a **world model** (a neural network that predicts what happens next)
- Then trains a policy **entirely inside the imagined world** (dreams)
- Dramatically more **sample-efficient** because you can generate unlimited imagined experience

**For Pokemon:** Interesting in theory — you could learn a Pokemon battle simulator in latent space. In practice, the real simulator (Pokemon Showdown) is fast enough that sample efficiency matters less than for robotics.

---

### 6. R-NaD (Regularized Nash Dynamics) — Game Theory Meets RL

This is what DeepMind used for **Stratego** (2022) — another imperfect-information game.

**How it works:**
- Instead of just maximizing win rate against the current opponent, it seeks a **Nash equilibrium** — a strategy that can't be exploited
- Uses regularization to stabilize the search for equilibrium
- Critical for **competitive games** where you need to be unexploitable, not just good against one opponent

**For Pokemon:** This is highly relevant philosophically. In VGC, if your agent always leads the same Pokemon, opponents will adapt. You want a mixed strategy (Nash equilibrium). However, R-NaD is complex and research-grade.

---

## What I'd Recommend for a Pokemon VGC Agent

The practical, proven path:

| Component | Approach |
|---|---|
| **Algorithm** | **PPO** (start here) |
| **Training regime** | **Self-play** with a league of past versions |
| **Environment** | Pokemon Showdown via [poke-env](https://github.com/hsahovic/poke-env) (Python Pokemon Showdown API) |
| **State representation** | Encode battle state (HP, status, field conditions, known opponent info) as a feature vector |
| **Action space** | Legal moves for both active Pokemon (with team preview selection as a separate phase) |
| **Framework** | [RLlib](https://docs.ray.io/en/latest/rllib/) or [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for PPO |

The **big challenges** specific to VGC:
1. **Team preview** (choosing 4 from 6) is a combinatorial selection problem — might need a separate policy or heuristic
2. **Simultaneous moves** for 2 Pokemon = large action space (need to handle this carefully)
3. **Imperfect information** — you can't see opponent's sets, which items they hold, etc.
4. **Reward shaping** — winning/losing is sparse; you'll likely want intermediate rewards (damage dealt, KOs, etc.)

Want me to start scaffolding out the project structure, or do you want to dig deeper into any of these algorithms first?

---

## User (Turn 2)

I remember learning about AlphaGo and how it mastered Go and thought that was cool. I have a lot of domain expertise in Pokemon, so I'm not daunted by simulating future states. Moreover, the action space doesn't seem that large -- for the information I control, each pokemon can switch or attack with one of four moves into two opposing slots. So one pokemon on one turn can take up to 10 different actions (2 from switch, then 4x2 maximally for attacking -- some moves may only have one available target, so this is a maximum). 10 actions per mon isn't a huge action space by any stretch, unless I'm missing a hidden world of actions?

From your synthesis, I'm interested in PPO and AlphaZero and R-NaD.

PPO -- I've heard of this before, seems like the current gold standard in RL
AlphaZero -- can best leverage my domain expertise in Pokemon, plus if it was powerful enough to learn as complex a game as Go, can't it do anything?
R-NaD -- playing Pokemon is sometimes more like "playing the man" than playing the game. If the agent has an exploitable, common play pattern, then it will get exploited. However, I want to start with VGC in a ladder-like format, where you only play one game, one opponent, so you can't get "learned" or exploited in the same way as if you played 100 games. But on the other hand, you may go up against a good player (who you can expect to make certain moves / run certain sets with a high degree of confidence) or a medium to bad player (who does suboptimal things that may end up working out). So striving for a Nash equilibrium (finding a strategy that works vs good and bad opponents) is also appealing to me

---

## Assistant (Turn 2)

You're thinking about this really well. Let me address your points and go deeper.

## Action Space — You're Mostly Right

Per-Pokemon, yes: up to ~10 actions (4 moves × up to 2 targets + up to 2 switches). But in VGC doubles, you pick actions for **both** active Pokemon simultaneously, so the joint action space is roughly:

**10 × 10 = ~100 joint actions per turn** (minus illegal combos like both switching to the same slot)

That's genuinely small. Go has ~250 legal moves per turn, Chess ~35, Starcraft has thousands. 100 is very tractable. Team preview is C(6,4) × lead orderings, also small. So action space is **not** your bottleneck — you're right to not worry about it.

The real complexity in Pokemon is the **information** problem, which brings me to the most important thing I need to flag:

---

## The Elephant in the Room: Imperfect Information

AlphaZero was designed for **perfect information** games — in Chess and Go, both players see the entire board. Pokemon is fundamentally **imperfect information**:

- You don't know the opponent's 4 movesets
- You don't know their items, abilities (sometimes), EVs, or natures
- You don't know what they'll click on the same turn as you (simultaneous moves)
- You don't know their back-line ordering or team preview reasoning

This matters enormously because AlphaZero's core loop — MCTS (Monte Carlo Tree Search) — assumes you can perfectly simulate forward from the current state. In Pokemon, **you can't**, because you don't know half the game state.

This doesn't mean search-based approaches are dead. It means you need **adaptations**. Let me go through all three algorithms with this in mind.

---

## PPO — Deep Dive

**Core idea:** Learn a policy (neural network) that maps battle states → action probabilities, by playing millions of games and adjusting the policy to do more of what worked.

**The actor-critic setup:**
- **Actor** (policy network): "Given this battle state, here's my probability distribution over the ~100 joint actions"
- **Critic** (value network): "Given this battle state, I think I have a 63% chance of winning"
- The critic helps reduce variance in training — instead of asking "did I win?", you ask "did I do better than expected?"

**The PPO trick (clipped objective):**
The policy gradient says "if an action led to a good outcome, increase its probability." The problem is you might increase it *way too much* on one lucky game. PPO clips the update: "you can shift this probability by at most ε (say 20%), no matter how good the outcome was." This is what makes it stable.

**For Pokemon specifically:**

Strengths:
- Handles imperfect information naturally — the policy just maps **observations** (what you can see) to actions. It doesn't need to know what's hidden.
- Simple to implement with libraries like Stable-Baselines3 or RLlib
- Self-play is straightforward: play against copies of yourself, train, repeat
- You can get a working agent relatively quickly

Weaknesses:
- **No explicit planning** — it's purely reactive. It learns patterns like "Incineroar + Fake Out = good turn 1" but doesn't think ahead like "if I Protect here, their Tailwind will run out next turn and then I'm faster"
- **Sample hungry** — needs millions of games, though Pokemon Showdown is fast enough for this
- Susceptible to **strategy collapse** in self-play (both agents converge to one narrow strategy). Mitigated by maintaining a "league" of past opponents.

---

## AlphaZero — Deep Dive (and Why It Needs Modification)

**Core loop:**
1. From the current position, run MCTS: simulate hundreds of possible futures
2. Use a neural network to evaluate leaf nodes ("who's winning here?") and suggest promising moves ("prior policy")
3. Pick the move that MCTS thinks is best
4. After the game, train the neural network on what MCTS discovered

**Why it's powerful for Go/Chess:**
MCTS + neural network guidance = deep, selective search. It doesn't look at every possibility — the neural network steers the search toward promising branches, and MCTS verifies them. This is why it can think 20+ moves ahead in Go despite the absurd branching factor.

**The imperfect information problem:**

In Chess, when MCTS simulates "what if I move my knight here?", it knows exactly what state that produces. In Pokemon, if you simulate "what if I use Thunderbolt on their Kyogre?", you need to know:
- Will they switch? To what?
- Will they Protect?
- Do they have Assault Vest? (hidden item)
- What's their SpDef investment? (hidden stats)

**Solutions — this is where your domain expertise matters:**

1. **Determinization / Information Set MCTS:** At each simulation, **sample** a plausible opponent state (random moveset, random item from likely candidates), then run normal MCTS as if you had perfect info. Repeat many times with different samples. Your domain expertise is gold here — you can build strong priors like "90% of Kyogre carry Water Spout, Choice Scarf or Mystic Water" instead of uniform random.

2. **Bayesian updating during battle:** As you observe opponent actions, narrow down their possible sets. "They didn't Water Spout turn 1 when it was the obvious play? Maybe they're Choice Specs with a different moveset, or they're Calm Mind."

3. **Player of Games (DeepMind, 2022):** This is literally the paper you want. It extends AlphaZero-style search to **imperfect information** games. It was demonstrated on both perfect-info games (Chess, Go) and imperfect-info games (Poker). It uses a concept called **public belief states** — representing what both players know to be common knowledge — and searches over those.

**For Pokemon specifically:**

Strengths:
- The search gives it **lookahead** — critical for VGC where positioning 2-3 turns out matters immensely
- Your domain expertise can massively improve the priors and determinization
- Naturally handles the "if I Protect this turn, what are my options next turn?" reasoning

Weaknesses:
- Significantly harder to implement than PPO
- Determinization can be brittle (if your opponent-modeling priors are wrong, the search is searching the wrong game)
- Simultaneous moves mean the game tree branches on *both* players' actions each turn — need to handle this with something like simultaneous-move MCTS

---

## R-NaD — Deep Dive

**The problem it solves:**

In Rock-Paper-Scissors, the Nash equilibrium is 33/33/33. If your RL agent trains via self-play with PPO, it might converge on "always play Rock" because at some point both copies happened to play Rock a lot, and once they're both playing Rock, there's no gradient to escape. R-NaD avoids this.

**How it works (simplified):**
- Maintain a policy
- Instead of maximizing "how much do I win?", minimize **exploitability** — how much could a perfect counter-strategy beat you?
- Uses regularization (entropy bonuses, averaging past policies) to keep the policy from oscillating (Rock → Paper → Scissors → Rock...)
- Converges toward Nash equilibrium rather than a fixed strategy

**Your ladder observation is astute.** In a single-game ladder format, you're right that individual opponents can't "learn" you. But:
- The metagame is a form of population-level exploitation. If your agent always leads Calyrex-Shadow + Flutter Mane, word gets around (or the ladder is full of people running Calyrex checks)
- A Nash-equilibrium agent would **mix** its team preview selections and leads unpredictably, making it robust against any opponent composition
- The "good player vs bad player" problem is real. A Nash strategy is **safe** against everyone but might leave wins on the table against bad players. A pure exploitative strategy (PPO) might win more against bad players but get punished by good ones.

**For Pokemon specifically:**

Strengths:
- Produces strategies that are robust and hard to counter
- Handles the "mixed strategy" nature of VGC naturally (randomizing leads, move selection)
- Theoretically optimal for competitive play

Weaknesses:
- Most complex to implement of the three
- Research-stage — fewer off-the-shelf implementations
- Pokemon is so complex that computing exact Nash equilibria is intractable; you're approximating

---

## My Honest Recommendation: A Phased Approach

| Phase | What | Why |
|---|---|---|
| **1** | PPO with self-play | Get something working, learn the plumbing (state encoding, action masking, reward shaping, Showdown integration). You'll have a bot that plays VGC within weeks. |
| **2** | Add search (AlphaZero-style with determinization) | Layer on lookahead using your domain priors for opponent sets. This is where your Pokemon knowledge becomes a competitive advantage. |
| **3** | Incorporate R-NaD ideas | Once you have a strong agent, harden it against exploitation by pushing toward mixed/Nash strategies. |

Each phase builds on the last. The state encoding, neural network architecture, and environment wrapper you build in Phase 1 carry forward. And you'll learn a ton about what matters in Pokemon RL by the time you reach Phase 2.

Want to start digging into the practical side — project structure, environment setup, state representation design? Or do you want to go even deeper on any of these algorithms?

The user has good domain expertise in Pokemon VGC and is thinking carefully about the problem. Let me address their points and go deeper into the three algorithms they're interested in.

On the action space point: They're right that per-Pokemon the action space isn't huge. But in doubles, you need to pick actions for BOTH Pokemon simultaneously, so it's roughly 10 × 10 = 100 joint actions per turn (minus illegal combos). That's still very manageable. Plus team preview (choosing 4 from 6) is C(6,4) = 15 options plus ordering matters for leads, so 15 × (ways to pick 2 leads from 4) = 15 × 6 = 90 options. Still not crazy.

Now I want to give them a thorough breakdown of PPO, AlphaZero, and R-NaD with specific considerations for Pokemon—especially how AlphaZero handles imperfect information and how these approaches deal with the Nash equilibrium problem in a competitive ladder setting.

---

## User (Turn 3)

AlphaZero's loop assumes you can perfectly simulate forward from the current state. But you can, can't you? When you're doing the rollout and simulating the future states, you start by rolling out randomly, then walking it back up to see what those decisions were good for, right? But I guess this is where the imperfect information comes in -- you unintentionally "information leak" backwards when you roll forwards. Like, the opponent doesn't know I have a Focus Sash, but in the rollout, they'll see that attacking into that pokemon 2x is better than attacking 1x (triggering and then overwhelming the sash, rather than just triggering it) and then prefer those paths, which is information you didn't know at the time of move selection. Is this how the imperfect information setup blocks this? Be more specific than just "you don't know the game state, so you can't simulate forward"

> [PPO has] No explicit planning

This is a big weakness for this setup if true. Pokemon can be solved by turn 5 of an 8 turn game, for example. Stalling out Rain / Sun / Tailwind / Light Screen / Reflect is a big part of the meta, and a native weakness to this kind of planning from algorithm selection doesn't bode very well.

> Information Set MCTS

This is sort of what it's like to do actual battles, if I'm reading this right. You see they lead Pelipper and Basculegion, and you have a pretty good guess of what their movesets are (knowing the meta). This is where knowing the meta is a plus, and it's exploitable by running off-meta stuff or being unpredictable. These are balanced by being bad in most cases, so as long as the agent isn't hyper narrowly focused, I think this is still okay. Get me a link to that Player of Games paper.

I think I want to start turning this into a repo. How do I do that in Cursor without losing this chat? If I click "Open Project" I would lose this chat I think. But I also want to start building a folder with some of this info in Documents/Personal/

---

## Assistant (Turn 3)

You're nailing the core issue. Let me get precise, find that paper, and then we'll set up your project without losing this chat.

## The Information Leakage Problem — Precisely

Your Focus Sash example is exactly right, and there's actually a formal name for this: **strategy fusion**.

Here's what happens step by step in a naive MCTS rollout in Pokemon:

**Turn state:** Your Garchomp (with hidden Focus Sash) is at full HP. Opponent has Flutter Mane active.

**In the MCTS simulation**, the game state is fully specified — Focus Sash is "known" to the simulation. When the tree expands the opponent's choices:

1. Branch A: Flutter Mane uses Moonblast on Garchomp → Sash triggers → Garchomp survives at 1 HP
2. Branch B: Flutter Mane uses Moonblast on your other Pokemon → Garchomp untouched

When MCTS backs up values, Branch A looks worse for *you* because the simulated opponent "sees" the Sash trigger and can follow up optimally next turn (double into Garchomp to finish it). The tree learns "my Garchomp is going to get double-targeted because they'll realize the Sash." But the **real** opponent might Moonblast Garchomp once, see it survive, be *surprised*, and not have positioned for the follow-up.

The corruption goes both directions:

- **Overestimates threats**: The search assumes opponents know your weaknesses (they'll "discover" your Sash, your weak coverage moves, your lack of Protect)
- **Underestimates bluffs/tricks**: Tricky plays that rely on opponent ignorance (like *not* Protecting when they expect you to, or running a surprise Tera type) get undervalued because the simulated opponent "knows" you won't Protect

The formal problem: in the search tree, **nodes that should be in different information sets** (opponent knows Sash vs. doesn't know Sash) get **fused into one node**. The backed-up value at that node is a corrupt average across worlds where the opponent has different knowledge. This is strategy fusion — the opponent's strategy across different information states gets fused together, giving them superhuman knowledge.

**This is fundamentally different from just "not knowing the state."** Even if you perfectly enumerate all possible opponent sets, the rollout still leaks YOUR hidden info to the simulated opponent, AND leaks the randomly-sampled opponent set to the simulated "you."

---

## On PPO and Planning

You're right to flag this. PPO learns **reactive patterns**, not explicit multi-turn plans. It *can* learn that "Tailwind is about to expire, so I should stall" — but only as an implicit pattern baked into the neural network weights after seeing that situation thousands of times. It won't "think through" the way you do when you count Tailwind turns.

That said, PPO agents with large enough networks and enough training *do* learn surprisingly deep implicit planning. OpenAI Five (Dota 2) planned teamfights multiple seconds ahead, all implicitly. But it took enormous compute. For Pokemon, where a turn-by-turn game is much shorter than a Dota match, this might be sufficient — but search-based approaches will be more sample-efficient at learning multi-turn tactics.

---

## Information Set MCTS — You've Got It

Yes, exactly. You're sampling "what could their team be?" from your knowledge of the meta, then running MCTS as if that sample is truth, then averaging across many samples. It's literally the mental model good players use: "They're probably Scarf, so I'll play as if they're Scarf, but hedge 20% for Specs."

The vulnerability to off-meta stuff is real and is the same vulnerability human players have. The mitigation is the same too: maintain some probability mass on unusual sets.

---

## Player of Games Paper

Let me find that for you.

The user has three things going on:

1. A deep technical question about WHY imperfect information breaks MCTS/AlphaZero - they have a specific hypothesis about information leakage that I should validate and expand on
2. A request for the Player of Games paper link
3. A practical question about starting a repo without losing this chat

Let me address all three. For #3, they can just start creating files right here - I can create files in their Documents/Personal/ directory. The chat won't be lost.

On the imperfect information point - they're EXACTLY right about the information leakage. Let me be very specific:

In MCTS, when you do a rollout from a node, both "players" in the simulation have access to the full game state. 

The simulated opponent "knows" things the real opponent wouldn't - this is the **strategy fusion** problem. Different information states collapse into a single node in the search tree, so the opponent's play uses information from all possible worlds rather than just their actual one. When values roll back up the tree, you're mixing branches where the opponent had completely different hidden information, which corrupts the minimax value at the root. Your Focus Sash example nails this - in the rollout, the simulated opponent can "see" whether Focus Sash will activate, but the real opponent can't.

This creates a systematic bias: MCTS overestimates threats because it assumes the opponent knows your weaknesses, while simultaneously underestimating your own opportunities because it assumes they know your tricks. I should look up the Player of Games paper to see how they addressed this, and I can start setting up the practical files in Documents/Personal/ right now since the chat persists in Cursor.

Here are the links:

- **Paper (arXiv):** [Player of Games (arXiv:2112.03178)](https://arxiv.org/abs/2112.03178)
- **Towards Data Science explainer:** [Why Player of Games Is Needed](https://towardsdatascience.com/why-player-of-games-is-needed-a01505c4bad7)

The paper is from December 2021 (published in journals in 2022). It's exactly the bridge between AlphaZero and imperfect-info games. It uses **public belief states** and **counterfactual regret minimization** at search nodes to avoid the strategy fusion problem.

---

## Starting Your Repo Without Losing This Chat

Good news: **you won't lose this chat.** Cursor chats persist in your workspace. You can create files and folders right here, right now — I have access to your filesystem. You don't need to "Open Project" to start building.

Let me create an initial project structure in `Documents/Personal/` for you. What would you like to call the project? Something like `pokemon-vgc-agent`? And I'll set up:

- A README capturing the key design decisions from this conversation
- An initial project structure for a Python RL project
- A research notes file with links to the papers we've discussed

Then later, when you're ready to work in it as a full project, you can open that folder in Cursor and this chat will still be accessible from your chat history.

---

## User (Turn 4)

Yes, call the folder that and set that repo up. Then open the folder in this chat, so that it's still open on the left hand side of the screen, and I can browse files, and I still have this agent chat. What I DON'T want is you to create files but have my Cursor's Primary Side Bar still show "You have not opened a folder. Open Folder"

---

## Assistant (Turn 4)

Let me create the project and then open it in your Cursor window.

The user wants me to:
1. Create the project folder and files in `Documents/Personal/pokemon-vgc-agent`
2. Open that folder in Cursor so it appears in the sidebar

Let me first create the directory structure, then I need to figure out how to open a folder in Cursor. I can use the `code` command (Cursor's CLI) to open a folder. Let me check if `cursor` command is available, or if it's `code`.

Actually, in Cursor, the CLI command is typically `cursor` to open a folder. But the user wants to keep this chat - opening a new folder in the same window should preserve the chat. Let me create the files first, then open the folder.

That should have opened `pokemon-vgc-agent` in your Cursor window — you should now see the file tree in the sidebar with `src/`, `research/`, `tests/`, `configs/`, and all the project files. This chat should still be here.

Here's what I set up:

```
pokemon-vgc-agent/
├── .git/
├── .gitignore
├── README.md              # Project overview + phased architecture plan
├── pyproject.toml         # Python project config + pylint settings
├── requirements.txt       # Dependencies (SB3, RLlib, poke-env, torch)
├── research/
│   └── notes.md           # Everything from this conversation — algorithms,
│                            strategy fusion, papers, VGC observations
├── src/
│   └── __init__.py
├── tests/
│   └── __init__.py
└── configs/
```

The `research/notes.md` file captures the key takeaways from our discussion — the three algorithm phases, the strategy fusion problem, the imperfect information solutions, and the paper links (PPO, AlphaZero, Player of Games, R-NaD/DeepNash).

Can you confirm the sidebar is showing the project? If it opened in a new window instead, let me know and we'll fix it. Once you're set, we can keep going — whether that's diving into the Player of Games paper, designing the state representation, or wiring up `poke-env`.

---

## User (Turn 5)

It's not showing it in the sidebar

---

## User (Turn 6)

Some key things I want to consider before diving into this are:

- How to collect data to generate effective priors
- How to simulate the "rollouts" truthfully

Collecting data -- there's a ladder on Smogon for this format, and users will sometimes upload the replays after a game. I could query this somehow and filter to only replays above a certain elo to only record high-level opponents.

Pro: regular stream of current data, following meta trends, somewhat easy integration (all data centralized in showdown)
Con: may bias towards the same players if not enough games get uploaded, can't tell full sets from replays, only what moves were used

Alt method -- community tournaments can sometimes collect hundreds of entrants, each of whom must submit their full team list. I copy this and positively weight winners with extra "existing" entries in the priors.

Pro: can tell full sets, not as strongly biased to the same players
Con: data comes inconsistently, not as easy to find (spread across different tournament organizers, rather than centralized in pokemon showdown)

I'd like to start with a POC using a single tournament's worth of data, and later we can figure out the problem of a recurring, current data pipeline.

How to simulate the rollout effectively -- I have no clue. I really really really don't want to have to replicate the pokemon battle rules here, as a single mistake in replicating it can lead to incorrect agent learning when transferred over to the real game. But I want to be able to do this setup a bunch of times:

- load a specific mid-battle state, e.g. Protect just used, move X disabled, 3 turns of rain, pokemon at 57% HP, attack lowered one stage, etc
- select a hypothetical set of moves / switches for both players
- execute these with real, correct outcomes
- repeat 2 & 3 many many times over for the same initial mid-battle state

Can poke-env do this? I trust Pokemon Showdown's battle engine, but can I "restart from a saved state" easily in this?

---

## Assistant (Turn 6)

*[Tools: WebSearch, WebSearch, WebSearch]*

These are both critical questions. The simulation one especially is an architectural decision that will shape the entire project. Let me research what's actually possible with poke-env and Showdown's sim.

The user is asking two important questions:

1. Data collection for priors (they have good ideas already, want to start with tournament data)
2. Battle simulation for rollouts - specifically, can they save/restore mid-battle states in Pokemon Showdown via poke-env?

The second question is critical and I need to research it carefully. Let me think about what I know and then search for specifics.

Pokemon Showdown is a JavaScript/TypeScript battle simulator. poke-env is a Python wrapper that connects to a Showdown server. The key question is: can you snapshot a battle state and restore it?

Looking at the actual architecture, poke-env communicates with Showdown via WebSocket, which means you can't easily fork or restore battle states mid-simulation. However, there's a more promising path: the `@pokemon-showdown/sim` package lets you use the battle engine directly as a library without needing the server, giving you direct access to the Battle class and its internal state for manipulation.

For MCTS to work effectively, I need to be able to clone battle states, apply moves to those clones, and repeat this thousands of times per turn—something that's fundamentally incompatible with a WebSocket-based approach due to latency. A Python reimplementation like `poke-engine` could work, but the most practical solution is probably using the sim package directly in JavaScript or TypeScript where I can serialize and restore Battle objects as needed.

*[Tools: WebFetch, WebFetch]*

This is great — there are real options here, each with tradeoffs. Let me also check the key details on Showdown's state serialization and poke-engine's completeness.

Excellent research results. Let me also look at the poke-engine repo and the Showdown serialization issue more carefully to give the user a clear picture.

Let me also search for the Showdown state serialization specifically.

*[Tools: WebFetch]*

*[Tools: WebFetch]*

*[Tools: Read(notes.md)]*

Okay, this is a really important finding. Let me lay out the landscape clearly.

## The Simulation Problem: Your Options

You're right to not want to reimplement the battle engine. Here's what actually exists:

### Option 1: Pokemon Showdown's Sim Directly (TypeScript)

This is the gold standard — it IS the correct engine. And crucially, **it supports exactly what you want**:

```typescript
// Clone a mid-battle state
const json = battle.toJSON();

// Restore it later (as many times as you want)
const clone = Battle.fromJSON(json);
```

This was [confirmed working](https://github.com/smogon/pokemon-showdown/issues/8105) by Showdown's creator (Zarel). There's also a clean npm package [`@pkmn/sim`](https://www.npmjs.com/package/@pkmn/sim) that extracts just the simulator without the server. It supports doubles, VGC, all generations, all mechanics.

**Pro:** Perfectly correct. State save/restore built in. Actively maintained by the Showdown team.
**Con:** It's TypeScript. Your RL code will be Python. You need a bridge.

### Option 2: poke-engine (Rust + Python bindings)

This is pmariglia's engine, purpose-built for searching through Pokemon states. It has MCTS, expectiminimax, state serialization, apply/undo instructions — exactly the primitives you'd want.

**Dealbreaker: it's singles only, gens 4-8.** No doubles, no VGC. And the author explicitly says it's "not a perfect engine" — it captures competitive essentials but is less complete than Showdown. So this fails both your correctness requirement and your format requirement.

### Option 3: poke-env (Python ↔ Showdown WebSocket)

poke-env talks to a Showdown server over WebSocket. Great for playing games. **Cannot save/restore states.** Someone opened an [issue requesting exactly this](https://github.com/hsahovic/poke-env/issues/190) and it was never implemented. Each battle is a one-way stream — you can't fork it.

**Use for:** Playing real games on ladder, training PPO (Phase 1).
**Not usable for:** MCTS rollouts.

### The Practical Architecture: Hybrid

Given all this, I think the right design is:

| Component | Tool | Language |
|---|---|---|
| **Playing real games** (ladder, self-play for PPO) | poke-env → Showdown server | Python |
| **MCTS rollouts** (Phase 2 search) | `@pkmn/sim` directly | TypeScript, called from Python |
| **RL training** (PPO, neural networks) | PyTorch / SB3 | Python |

The bridge between Python and the TS sim is the key engineering challenge. Practical options:

1. **Node.js subprocess** — Python spawns a Node process, sends serialized states over stdin/stdout, gets results back. Simple, a bit slow per-call, but you can batch.
2. **Local HTTP/gRPC server** — A thin Node.js service wrapping `@pkmn/sim` that your Python code calls. Clean API boundary, easy to parallelize.
3. **Write the search in TypeScript too** — Run MCTS entirely in Node using the sim directly, only cross to Python for neural network evaluation. This is actually the fastest option since the tight loop (clone state → apply move → evaluate) stays in one process.

Option 3 is the most performant for MCTS since you're cloning and simulating thousands of states per turn decision. Crossing the Python↔JS boundary thousands of times per move would be slow. Better to have the search loop in JS, calling back to Python only for neural net evaluations (which happen at leaf nodes, much less frequently).

---

## Data Collection

Your two methods are both solid. A few additional things worth knowing:

**Showdown publishes usage statistics** at [smogon.com/stats](https://www.smogon.com/stats/) — monthly dumps of move usage, item usage, ability usage, teammate correlations, per format, stratified by rating. This is essentially a pre-built prior distribution for "what sets do people run?" It won't give you full 4-move sets, but it gives you marginal probabilities (e.g., "72% of Incineroar carry Fake Out, 65% carry Flare Blitz") that you can use to sample plausible complete sets.

For your POC with tournament data, that's a great starting point since you get **complete team sheets** — full sets, EVs, everything. You can build exact set distributions rather than marginal move probabilities.

---

Let me update the research notes with these findings.

Now I have a very clear picture. Let me synthesize this for the user.

There are essentially three options for the simulation engine:

1. **Pokemon Showdown's sim directly** (TypeScript/JavaScript) — `Battle.toJSON()` / `Battle.fromJSON()` works. This is the most correct engine. But it's JS, not Python.

2. **poke-engine by pmariglia** (Rust with Python bindings) — Built specifically for searching through Pokemon states. Has MCTS built in. Supports state serialization, applying/undoing instructions. BUT: it's singles only (gens 4-8), not doubles/VGC. 

That's a dealbreaker for VGC work.

3. **poke-env** (Python ↔ Showdown WebSocket) — Can play games but can't clone/fork states. Not suitable for MCTS rollouts.

So I'm looking at a few practical paths forward: using Showdown's sim in JavaScript for rollouts since it handles state cloning correctly, forking poke-engine to add doubles support which would be substantial work, writing a custom Python engine which the user wants to avoid, or going hybrid with poke-env for real games and Showdown's sim for the search component.

The hybrid approach seems most viable—using poke-env to connect to Showdown and play on ladder, then serializing the battle state into separate Node.js instances of Showdown's sim for MCTS rollouts. The main friction is bridging Python and JavaScript, whether through child processes, an HTTP wrapper, or just writing everything in TypeScript. I'm also noticing that `@pkmn/sim` on npm is a clean extraction of just the simulator, and Showdown's battle object has JSON serialization methods that actually work.

One critical thing though—poke-engine only handles singles, so if the user wants VGC doubles support, that's a blocker. And for the data collection side, Showdown's own usage statistics could provide useful priors for the model.

*[Tools: StrReplace]*

Notes updated. Here's the key takeaway on the simulation question:

**You can do exactly what you described** — load a mid-battle state, fork it, run hypothetical moves, get correct outcomes — using Pokemon Showdown's own sim (`@pkmn/sim`). The `Battle.toJSON()` / `Battle.fromJSON()` round-trip is the mechanism. No reimplementation needed. You get the real, correct Showdown engine.

The architectural cost is that the sim is TypeScript, your RL is Python, so you need a bridge. The cleanest design is to keep the MCTS search loop in Node.js (where cloning/simulating is fast and in-process) and only call back to Python when you need a neural network evaluation at a leaf node. That minimizes cross-language overhead.

Want to start exploring `@pkmn/sim` and prototype the state save/restore cycle? Or tackle the data collection side first with the tournament team sheets?

---

## User (Turn 7)

The format I want to optimize for here is called Pokemon Champions. It's a newly released game for the Switch, and one of the formats is: (1) limited dex, (2) limited movepool, (3) double battle, (4) bring 6, pick 4. This game is new, so the meta is still early, but a key element of this game is that IVs don't exist (all assumed max 31) and the EVs are slightly different. Do some research about Pokemon Champions and the double battle format and create a cursor rule explaining this format for future coding agents to understand

---

## Assistant (Turn 7)

*[Tools: WebSearch, WebSearch]*

*[Tools: WebSearch, WebSearch, WebSearch]*

*[Tools: WebSearch, WebSearch]*

*[Tools: WebSearch, WebSearch]*

*[Tools: WebSearch, WebFetch]*

*[Tools: WebSearch, WebSearch]*

Now I have a comprehensive understanding of Pokemon Champions. Let me also check if Showdown actually supports this format, since that's critical for the simulation engine.

*[Tools: WebFetch]*

*[Tools: Read(SKILL.md)]*

*[Tools: Shell(`mkdir -p /Users/andytaylor/Documents/Personal/pokemon-vgc-ag...`)]*

*[Tools: Write(pokemon-champions-format.mdc)]*

*[Tools: Read(README.md)]*

*[Tools: StrReplace]*

*[Tools: StrReplace]*

Done. Here's a summary of what I found and what I created:

## Key Findings

**Pokemon Champions format:**
- ~195 fully-evolved Pokemon, 59 Mega Evolutions, both Mega and Tera available (one each per team, can't stack on the same mon)
- Stat Points (66 total, 32 max per stat) replace EVs. IVs gone entirely (all 31). Formula is otherwise the same as mainline.
- 66 stat points gives you 1 more point of budget than classic EVs — a subtle but real difference for calcs.

**Showdown support:** Showdown appears to support the format (the community is already playing it), but I couldn't confirm the exact format ID (e.g. `gen10vgc2026`). You may need to check `play.pokemonshowdown.com` directly to confirm the format string for poke-env / `@pkmn/sim`.

**Simulation:** The critical constraint holds — `@pkmn/sim` with `Battle.toJSON()` / `Battle.fromJSON()` is still the path for rollouts, since the game uses the same core battle engine that Showdown implements.

## What I Created

The Cursor rule is at `.cursor/rules/pokemon-champions-format.mdc` with `alwaysApply: true` — every future agent conversation in this project will have full context on the format, stat system, gimmick rules, action space, imperfect information boundaries, and simulation constraints. I also updated the README to reflect Pokemon Champions specifically.

Where to next?

---

## User (Turn 8)

Tera is not available in Champions yet, nor is terrain (I think). Remove this simulation engine section from the rule, that's out of scope. Remove the action space and imperfect information sections from the rule too. I want this rule to focus solely on what distinguishes Pokemon Champions from other formats. Create another rule that describes the project overview, where I think the sim engine and the action space and the imperfect information sections belong

---

## Assistant (Turn 8)

*[Tools: Read(pokemon-champions-format.mdc)]*

*[Tools: Write(pokemon-champions-format.mdc), Write(project-overview.mdc)]*

Done. Two rules now:

- **`.cursor/rules/pokemon-champions-format.mdc`** — focused purely on what makes Pokemon Champions distinct: the limited dex, stat points replacing EVs/IVs, Mega Evolution only (no Tera), and key doubles mechanics. Removed terrain, Tera references, action space, imperfect info, and simulation engine.

- **`.cursor/rules/project-overview.mdc`** — the RL project itself: phased architecture, action space, imperfect information / strategy fusion, and simulation engine constraints. This is where the "how we're building it" context lives.

Both are `alwaysApply: true` so every future agent chat in this project gets both contexts automatically.

---

## User (Turn 9)

I want the final solution to include some amount of MCTS rollout -- is there therefore any value in initially pursuing PPO? Do they really work in parallel / conjunction with each other? How does AlphaZero use MCTS -- is it with another RL algorithm like PPO? Why are you saying in the overview rule to start with PPO then move to search? Does my final solution (which MUST involve MCTS search) gain any ground by starting with PPO?

---

## Assistant (Turn 9)

You're right to push on this. Let me be honest — my phased recommendation was somewhat misleading. Let me clarify how AlphaZero actually works, because it answers all your questions at once.

## How AlphaZero Actually Trains

AlphaZero does **not** use PPO. It doesn't use any traditional RL algorithm. Here's the actual loop:

1. **Play a game** using MCTS guided by a neural network. At each turn, run ~800 MCTS simulations. The neural network has two outputs:
   - **Policy head**: "which moves look promising?" (guides which MCTS branches to explore)
   - **Value head**: "who's winning from this position?" (evaluates leaf nodes so you don't need to roll out to the end of the game)

2. **Record training data** from the game. For every position you encountered, store:
   - The board state
   - The MCTS visit counts (which moves MCTS actually favored after thinking — this is the "improved" policy)
   - Who won the game

3. **Train the neural network** via plain supervised learning:
   - Policy loss: make the network's policy match the MCTS visit distribution
   - Value loss: make the network's value prediction match the game outcome

4. **Repeat.** The better the network gets, the better MCTS plays, which generates better training data, which improves the network.

That's it. There is no PPO anywhere in this loop. The "RL" in AlphaZero is the self-play data generation — MCTS is the improvement operator, and the network just learns to distill what MCTS discovered.

## So Where Did My PPO Recommendation Come From?

The shared work between PPO and AlphaZero is:
- Environment wrapper (connecting to the battle sim)
- State encoding (turning a battle into a tensor)
- Action masking (filtering illegal moves)
- Neural network architecture (the encoder that processes battle state)

My reasoning was "build these with PPO because it's simpler, then swap in MCTS." But that's a weak argument because:

- **You build all that infrastructure anyway** as part of AlphaZero — it doesn't require PPO as a stepping stone
- **The PPO training loop itself gets thrown away** — policy gradient code, GAE advantage estimation, clipping — none of it carries over
- **The state representation may actually differ** — PPO operates on observations (imperfect info only), while AlphaZero's network in a determinized setup might see a "completed" state with sampled opponent info
- **Time on PPO is time not on the target system**

## Does Your Final Solution Gain Ground from PPO?

**No, not meaningfully.** If you're committed to MCTS search, go straight for it. The real engineering challenges are:

1. The `@pkmn/sim` integration with state cloning (this is MCTS-specific, PPO doesn't touch it)
2. Handling imperfect information in the search tree (determinization, information sets)
3. The Python ↔ JS bridge for neural net evaluation during search
4. The self-play + training loop where MCTS generates training data

None of those get easier by doing PPO first.

## When PPO IS Useful Alongside Search

There is one legitimate combined architecture: **use PPO (or any policy gradient method) to train the neural network, but use MCTS at inference time only.** This is sometimes called "search at test time." You train a fast reactive policy with PPO, then at game time, you augment it with search for better decisions. But this is a different design than AlphaZero, and arguably weaker — AlphaZero's whole insight is that training against MCTS-improved targets produces a better network than training against raw policy gradient targets.

---

Want me to update the project overview rule to reflect this? The phased plan should probably be restructured around building the AlphaZero-style system directly, with milestones like "environment working → state cloning working → MCTS producing legal games → neural network integration → self-play training loop."

This is a really good question. The user is pushing back on the phased approach and asking whether PPO is actually useful if the end goal is MCTS-based search. Let me think through this carefully.

AlphaZero's architecture:
- AlphaZero does NOT use PPO. It uses MCTS + a neural network.
- The neural network has two heads: a policy head (predicts move probabilities) and a value head (predicts win probability).
- MCTS runs many simulations, guided by the neural network's policy (to decide which branches to explore) and value (to evaluate leaf nodes).

- After MCTS completes, the visit counts become the training target for the policy network, creating a loop where the network guides search, search produces better move probabilities, and the network learns from those improved probabilities.

The training is just supervised learning on MCTS's visit distribution and game outcomes — not PPO or traditional RL, but rather self-play plus search generating training data that the network learns from. So the user's instinct is right: if the final system uses MCTS, PPO isn't the right approach.

These are fundamentally different paradigms. PPO trains a network to directly output actions via policy gradients, while AlphaZero uses the network to guide MCTS, which then produces the actions. They're alternatives, not something you'd combine.

Starting with PPO does have some practical merit though — the environment wrapper, state encoding, action masking, and reward function all need to be built regardless, and PPO is simpler to get working quickly to validate that infrastructure. The network architecture you design carries over to AlphaZero too, and PPO is much easier to debug without worrying about search trees. But there's a tradeoff since you'd be building infrastructure that doesn't directly transfer to the final approach.

The real question is whether that validation is worth the detour. If you already know you want MCTS search, you could build those shared components directly as part of the AlphaZero pipeline instead. The state representation might even differ between the two approaches — PPO works with observations while AlphaZero might see a completed state in a determinized setup. So the honest answer is that PPO is mostly a detour if the target is already clear, though there are hybrid approaches like Expert Iteration that blur the lines between the two paradigms. I should clarify this for the user directly rather than burying the correction in implementation details. The phased approach I suggested earlier was misleading—PPO and AlphaZero aren't sequential steps in the same pipeline, they're fundamentally different training paradigms. What they'd actually share is the environment simulator and the neural network architecture, but the learning algorithms themselves are distinct. I need to be upfront about this misconception.

---

## User (Turn 10)

Yes, update the rule

---

## Assistant (Turn 10)

*[Tools: Write(project-overview.mdc)]*

*[Tools: Read(README.md)]*

Also updating the README to stay consistent.

*[Tools: StrReplace]*

Updated both files. The project overview rule now reflects the actual AlphaZero training loop (MCTS generates training data → network learns via supervised learning → repeat), with clear milestones building toward that directly. No more PPO detour. README is consistent too — also fixed the Tera reference there.

---
