# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for iteration in range(self.iterations):
            # Create a copy of the current values (batch update)
            newValues = util.Counter()
            
            # For each state
            for state in self.mdp.getStates():
                # If terminal state, value stays 0
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                else:
                    # Compute the maximum Q-value over all possible actions
                    qValues = []
                    for action in self.mdp.getPossibleActions(state):
                        # Compute Q-value for this action
                        qValue = 0
                        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                            reward = self.mdp.getReward(state, action, nextState)
                            qValue += prob * (reward + self.discount * self.values[nextState])
                        qValues.append(qValue)
                    
                    # Set the value to the max Q-value
                    if qValues:
                        newValues[state] = max(qValues)
                    else:
                        newValues[state] = 0
            
            # Update values after processing all states (batch update)
            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + self.discount * self.values[nextState])
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        possibleActions = self.mdp.getPossibleActions(state)
        
        # If no legal actions, return None
        if not possibleActions:
            return None
        
        # Compute Q-values for all actions
        bestAction = None
        bestQValue = float('-inf')
        
        for action in possibleActions:
            qValue = self.computeQValueFromValues(state, action)
            if qValue > bestQValue:
                bestQValue = qValue
                bestAction = action
        
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Get the list of states
        states = self.mdp.getStates()
        numStates = len(states)
        
        # Iterate for the specified number of iterations
        for iteration in range(self.iterations):
            # Get the state to update in this iteration (cyclic)
            stateIndex = iteration % numStates
            state = states[stateIndex]
            
            # Skip terminal states
            if self.mdp.isTerminal(state):
                continue
            
            # Compute the maximum Q-value over all possible actions
            possibleActions = self.mdp.getPossibleActions(state)
            
            if possibleActions:
                # Compute Q-values for all actions
                maxQValue = float('-inf')
                for action in possibleActions:
                    qValue = 0
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        reward = self.mdp.getReward(state, action, nextState)
                        qValue += prob * (reward + self.discount * self.values[nextState])
                    maxQValue = max(maxQValue, qValue)
                
                # Update this state's value
                self.values[state] = maxQValue
            else:
                # No legal actions: value stays 0
                self.values[state] = 0

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Step 1: Compute predecessors of all states
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = set()
        
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)
        
        # Step 2: Initialize priority queue
        priorityQueue = util.PriorityQueue()
        
        # Step 3: For each non-terminal state, compute initial priority
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                # Compute the highest Q-value across all possible actions
                possibleActions = self.mdp.getPossibleActions(state)
                if possibleActions:
                    maxQValue = float('-inf')
                    for action in possibleActions:
                        qValue = 0
                        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                            reward = self.mdp.getReward(state, action, nextState)
                            qValue += prob * (reward + self.discount * self.values[nextState])
                        maxQValue = max(maxQValue, qValue)
                else:
                    maxQValue = 0
                
                # Compute diff
                diff = abs(self.values[state] - maxQValue)
                
                # Push with negative priority (min heap, so negative for max heap behavior)
                priorityQueue.push(state, -diff)
        
        # Step 4: Main iteration loop
        for iteration in range(self.iterations):
            # If priority queue is empty, terminate
            if priorityQueue.isEmpty():
                break
            
            # Pop a state from the queue
            state = priorityQueue.pop()
            
            # Update the value of s (if not terminal)
            if not self.mdp.isTerminal(state):
                possibleActions = self.mdp.getPossibleActions(state)
                if possibleActions:
                    maxQValue = float('-inf')
                    for action in possibleActions:
                        qValue = 0
                        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                            reward = self.mdp.getReward(state, action, nextState)
                            qValue += prob * (reward + self.discount * self.values[nextState])
                        maxQValue = max(maxQValue, qValue)
                    self.values[state] = maxQValue
                else:
                    self.values[state] = 0
            
            # For each predecessor of s
            for predecessor in predecessors[state]:
                if not self.mdp.isTerminal(predecessor):
                    # Compute the highest Q-value for the predecessor
                    possibleActions = self.mdp.getPossibleActions(predecessor)
                    if possibleActions:
                        maxQValue = float('-inf')
                        for action in possibleActions:
                            qValue = 0
                            for nextState, prob in self.mdp.getTransitionStatesAndProbs(predecessor, action):
                                reward = self.mdp.getReward(predecessor, action, nextState)
                                qValue += prob * (reward + self.discount * self.values[nextState])
                            maxQValue = max(maxQValue, qValue)
                    else:
                        maxQValue = 0
                    
                    # Compute diff
                    diff = abs(self.values[predecessor] - maxQValue)
                    
                    # If diff > theta, update the priority queue
                    if diff > self.theta:
                        priorityQueue.update(predecessor, -diff)


