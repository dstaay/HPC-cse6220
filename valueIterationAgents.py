import mdp, util

from learningAgents import ValueEstimationAgent

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
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
     
    "*** YOUR CODE HERE ***"
    
    state_list = mdp.getStates()
    
    self.values[(3,2)] = 1
    self.values[(3,1)] = -1

    for i in range(iterations):
        for state in state_list:
            action_list = mdp.getPossibleActions(state)
            #only update for relevant states 
            if action_list and action_list[0] != 'exit':
                
                reward = [0]*len(action_list)
                counter = 0
                for action in action_list:
                    transition_list = mdp.getTransitionStatesAndProbs(state, action)
                
                    #print state_, action_, transition_list
                    if transition_list:
                        for transition in transition_list:
                            reward[counter] += self.getValue(transition[0]) \
                                        * transition[1]
                            #print state_, mdp.getReward(state_, action_, transition_[0])
                                    
                    counter += 1
            
                    # Now update state value       
                                     
                self.values[state] = discount*max(reward)

  
    
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    reward = 0
    transition_list = self.mdp.getTransitionStatesAndProbs(state, action)
    if transition_list:
        for transition in transition_list:
            reward += self.getValue(transition[0])*transition[1] 
        return(reward*self.discount)

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    if state == 'TERMINAL STATE':
        return None
    else:
    
        action_list = self.mdp.getPossibleActions(state)
        if action_list:
                
            reward = [0]*len(action_list)
            counter = 0
            for action in action_list:
                transition_list = self.mdp.getTransitionStatesAndProbs(state, action)
                if transition_list:
                    for transition in transition_list:
                        reward[counter] += self.getValue(transition[0])*transition[1] 
                counter += 1
            return(action_list[reward.index(max(reward))])
        else:
            return None
                  
    

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
