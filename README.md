# Project 3: Reinforcement Learning

UC Berkeley CS 188 Pacman AI project focused on reinforcement learning. Students implement value iteration and Q-learning agents, then apply them to Gridworld, a simulated robot crawler, and Pacman.

## Student Files to Edit

- `valueIterationAgents.py` - Value iteration agent for Gridworld
- `qlearningAgents.py` - Q-learning agents for Gridworld, Crawler, and Pacman
- `analysis.py` - Answers to analysis questions (parameter tuning)

## Running

### Gridworld (manual control)

```bash
python3 gridworld.py -m
```

### Autograder

```bash
python3 autograder.py
```

Grade a specific question:

```bash
python3 autograder.py -q q1
```

### Crawler

```bash
python3 crawler.py
```

### Pacman with Q-learning

```bash
python3 pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
```

## Key Concepts

- **Value Iteration** (offline planning): computes optimal policy from a known MDP
- **Q-Learning** (online learning): learns action values from experience without a model
- **Epsilon-Greedy**: exploration strategy balancing random exploration with learned policy
- **Approximate Q-Learning**: uses feature extractors to generalize across states

## Project Structure

| File | Description |
|------|-------------|
| `mdp.py` | MDP interface definition |
| `environment.py` | Environment interface |
| `learningAgents.py` | Base classes for learning agents |
| `featureExtractors.py` | Feature extractors for approximate Q-learning |
| `gridworld.py` | Gridworld MDP implementation and main driver |
| `crawler.py` | Crawler robot simulation |
| `pacman.py` | Pacman game engine |
| `autograder.py` | Project autograder |
| `test_cases/` | Test cases for each question |

## Implementation Summary

### Question 1: Value Iteration Agent
**File**: `valueIterationAgents.py`

Implemented batch-style value iteration that computes the optimal policy for a known MDP. Key methods:
- **`runValueIteration()`**: Executes value iteration for a specified number of iterations
  - Uses batch updates: all states' values computed from previous iteration's values
  - Handles terminal states (value = 0) and states with no legal actions
  - Iterates: V(s) ← max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]

- **`computeQValueFromValues(state, action)`**: Computes Q(s,a) from learned values
  - Q(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]

- **`computeActionFromQValues(state)`**: Extracts greedy policy
  - Returns action maximizing Q-value; handles ties and terminal states

**Result**: Passes all tests. Agent computes correct values and policies matching expected results.

---

### Question 2: BridgeGrid Parameter Tuning
**File**: `analysis.py` - `question2()`

Determined optimal MDP parameters for agent to cross the bridge in BridgeGrid:
- **Answer**: `discount=0.9, noise=0.0`
- **Rationale**: Reducing noise from 0.2 to 0.0 makes the narrow bridge path reliable and eliminates the risk of falling into the cliff. With deterministic actions, the long-term payoff of the distant high-reward terminal state justifies the journey across.
- **Why not increase discount**: Discount factor of 0.9 already values distant rewards sufficiently; the bottleneck is the stochastic uncertainty of the path.

---

### Question 3: DiscountGrid Policy Optimization (3a-3e)
**File**: `analysis.py` - `question3a()` through `question3e()`

Tuned discount, noise, and living reward parameters to produce five different optimal policies:

| Scenario | Discount | Noise | Living Reward | Policy |
|----------|----------|-------|---------------|--------|
| **3a**: Close + Risk | 0.3 | 0.0 | 0 | Prefers +1 exit via cliff |
| **3b**: Close + Avoid | 0.3 | 0.3 | 0 | Prefers +1 exit via safe path |
| **3c**: Distant + Risk | 0.99 | 0.0 | 0 | Pursues +10 via cliff |
| **3d**: Distant + Avoid | 0.99 | 0.3 | 0 | Pursues +10 via safe path |
| **3e**: Never Terminate | 0.99 | 0.0 | 0.1 | Avoids all exits indefinitely |

Key insights:
- **Discount factor**: Controls time horizon (low → myopic, high → far-sighted)
- **Noise**: Controls risk tolerance (high noise makes unsafe paths expensive)
- **Living reward**: Makes non-terminal states valuable (prevents convergence to any terminal state)

---

### Question 4: Asynchronous Value Iteration
**File**: `valueIterationAgents.py` - `AsynchronousValueIterationAgent`

Implemented cyclic asynchronous value iteration that updates one state per iteration:
- **Algorithm**: For each iteration i, update state at position (i mod numStates)
- **Updates are immediate** (not batched), so newer values are used in subsequent updates
- **Skips terminal states** (continue to next iteration)
- **More efficient** than batch updates for large state spaces
- **Convergence**: Typically faster than batch value iteration for this problem

**Result**: Passes all tests with efficient convergence.

---

### Question 5: Prioritized Sweeping Value Iteration
**File**: `valueIterationAgents.py` - `PrioritizedSweepingValueIterationAgent`

Implemented intelligent prioritized sweeping that focuses updates on high-error states:

**Algorithm**:
1. Compute predecessors of all states (backward reachability)
2. Initialize priority queue with all non-terminal states
   - Priority = -|V(s) - max_a Q(s,a)| (negative for min-heap)
3. For each iteration:
   - Pop highest-priority state
   - Update its value
   - For each predecessor, check if error > θ (theta threshold)
   - Re-insert predecessor if threshold exceeded

**Efficiency**: Only updates states with significant errors, avoiding wasted computation on converged states.

**Result**: Passes all tests. Converges faster than both batch and asynchronous value iteration.

---

### Question 6: Q-Learning Agent (Core)
**File**: `qlearningAgents.py` - `QLearningAgent`

Implemented tabular Q-learning agent that learns from experience without a model:

- **`__init__`**: Initializes `self.qValues = util.Counter()` for Q-value storage
  - Counter provides default value 0.0 for unseen state-action pairs

- **`getQValue(state, action)`**: Returns learned Q-value
  - Automatically returns 0.0 for unseen pairs

- **`computeValueFromQValues(state)`**: Returns V(s) = max_a Q(s,a)
  - Returns 0.0 for terminal states

- **`computeActionFromQValues(state)`**: Selects best action with **random tie-breaking**
  - Finds all actions with maximum Q-value and randomly selects one
  - Handles terminal states (returns None)

- **`getAction(state)`**: Implements **epsilon-greedy exploration**
  - With probability ε: select random legal action
  - With probability 1-ε: select best policy action

- **`update(state, action, nextState, reward)`**: Q-learning update rule
  - Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
  - Off-policy: learns optimal policy while following exploratory policy

**Result**: Passes all tests. Correctly learns Q-values for gridworld domains.

---

### Question 7: Epsilon-Greedy Action Selection
**File**: `qlearningAgents.py` - `QLearningAgent.getAction()`

Completed the implementation with proper epsilon-greedy exploration:
- Uses `util.flipCoin(self.epsilon)` for probabilistic exploration
- Returns random legal action (any action, including unseen ones) with probability ε
- Returns best policy action with probability 1-ε
- Enables learn-while-exploring paradigm

**Result**: Passes all tests. Agent learns to win on smallGrid with 100% test success rate.

---

### Question 8: BridgeGrid Q-Learning Analysis
**File**: `analysis.py` - `question8()`

Analyzed whether any epsilon and learning rate combination can achieve >99% optimal policy success in 50 training episodes on BridgeGrid:

- **Answer**: `'NOT POSSIBLE'`
- **Reasoning**:
  - Low epsilon (0-0.1): Insufficient exploration to discover bridge path
  - High epsilon (>0.3): Too much random action, insufficient time to converge
  - 50 episodes: Far too few to both discover complex path AND learn optimal policy with >99% certainty
  - Problem is inherently difficult due to required exploration depth

---

### Question 9: Pacman Q-Learning Agent
**File**: `qlearningAgents.py` - `PacmanQAgent`

Applied Q-learning to Pacman game with optimized parameters:
- **Command**: `python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid`
- **Parameters**: ε=0.05, α=0.2, γ=0.8
- **Result**: 100% test win rate (exceeds 80% requirement)

Key features:
- Training: 2000 episodes with exploration (ε=0.05)
- Testing: 10 episodes with pure exploitation (ε=0, α=0)
- Learns effective policy to navigate gridworld and defeat ghosts

**Result**: Passes test with full credit (1/1). Agent successfully learns to play Pacman.

---

### Question 10: Approximate Q-Learning with Features
**File**: `qlearningAgents.py` - `ApproximateQAgent`

Implemented approximate Q-learning for scalability to larger domains:

- **`getQValue(state, action)`**: Computes Q(s,a) = w · f(s,a)
  - Dot product of weight vector with feature vector
  - Enables generalization across similar states

- **`update(state, action, nextState, reward)`**: Updates weights via gradient descent
  - Difference: δ = r + γ max_a' Q(s',a') - Q(s,a)
  - Weight update: w_i ← w_i + α·δ·f_i(s,a)
  - Same learning rate and discount factor from parent class

- **Feature extractors**:
  - **IdentityExtractor**: Each (state,action) → unique feature (equivalent to tabular Q-learning)
  - **SimpleExtractor**: Abstracts features like ghost distance, food distance, bias
  - Enables learning on larger domains (mediumGrid, mediumClassic)

**Advantages**:
- Generalizes across states through shared features
- Scales to larger domains beyond gridworld
- Maintains compatibility with parent Q-learning methods
- Works with any custom feature extractor

**Result**: Passes all tests (3/3). Correctly learns feature weights and generalizes to new states.

---

## Attribution

The Pacman AI projects were developed at UC Berkeley by John DeNero and Dan Klein. Student-side autograding by Brad Miller, Nick Hay, and Pieter Abbeel. More info at http://ai.berkeley.edu.