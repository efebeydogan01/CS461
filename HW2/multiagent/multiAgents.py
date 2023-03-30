# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        pellets = currentGameState.getCapsules()

        foodDistance = min(manhattanDistance(newPos, foodPos) for foodPos in foodList) if foodList else 0
        ghostsNear = sum(manhattanDistance(newPos, ghostPos) <= 1 for ghostPos in successorGameState.getGhostPositions())
        
        pelletDistance = min(manhattanDistance(newPos, pelletPos) for pelletPos in pellets) if pellets else 0

        scaredGhostsNear = [scaredTime - manhattanDistance(newPos, ghostState.getPosition()) for ghostState, scaredTime in zip(newGhostStates, newScaredTimes) if scaredTime > 0]
        maxScaredGhostsNear = max(scaredGhostsNear) if scaredGhostsNear else 0
        return 10*successorGameState.getScore() - foodDistance - 1000*ghostsNear - 2*pelletDistance + 2000*maxScaredGhostsNear

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        move = self.value(gameState, self.index, 0, returnMove=True)
        return move
    
    def value(self, gameState: GameState, agentIndex, depth, returnMove=False):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(agentIndex)
        successors = (gameState.generateSuccessor(agentIndex, move) for move in legalMoves)

        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgentIndex == 0: 
            depth += 1
        values = [self.value(successor, nextAgentIndex, depth) for successor in successors]

        if agentIndex == 0:
            v = max(values)
        else:
            v = min(values)
        
        return v if not returnMove else legalMoves[values.index(v)]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        move = self.value(gameState, self.index, depth=0, a=float("-inf"), b=float("inf"), returnMove=True)
        return move
    
    def value(self, gameState: GameState, agentIndex, depth, a, b, returnMove=False):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        maxNode = agentIndex == 0
        v = float("-inf") if maxNode else float("inf")
        chosenMove = None
        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgentIndex == 0: 
            depth += 1

        for move in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, move)
            v_new = self.value(successor, nextAgentIndex, depth, a, b)

            if maxNode and v_new > v:
                v = v_new
                chosenMove = move
                if v > b: return v
                a = max(a, v)
            elif not maxNode and v_new < v:
                v = v_new
                chosenMove = move
                if v < a: return v
                b = min(b, v)
                
        return v if not returnMove else chosenMove

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        move = self.value(gameState, self.index, 0, returnMove=True)
        return move
    
    def value(self, gameState: GameState, agentIndex, depth, returnMove=False):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(agentIndex)
        successors = (gameState.generateSuccessor(agentIndex, move) for move in legalMoves)

        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgentIndex == 0: 
            depth += 1
        values = [self.value(successor, nextAgentIndex, depth) for successor in successors]

        if agentIndex == 0:
            v = max(values)
        else:
            v = sum(values) / len(values)
        
        return v if not returnMove else legalMoves[values.index(v)]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: penalty for higher food distance, huge penalty for ghosts at 1 unit distance, penalty for pellet distance, reinforcement for chasing scared ghosts
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    foodList = food.asList()
    pellets = currentGameState.getCapsules()

    foodDistance = min(manhattanDistance(pos, foodPos) for foodPos in foodList) if foodList else 0
    ghostsNear = sum(manhattanDistance(pos, ghostPos) <= 1 for ghostPos in currentGameState.getGhostPositions())
    
    pelletDistance = min(manhattanDistance(pos, pelletPos) for pelletPos in pellets) if pellets else 0
    
    scaredGhostsNear = [scaredTime - manhattanDistance(pos, ghostState.getPosition()) for ghostState, scaredTime in zip(ghostStates, scaredTimes) if scaredTime > 0]
    maxScaredGhostsNear = max(scaredGhostsNear) if scaredGhostsNear else 0

    return 10*currentGameState.getScore() - foodDistance - 1000*ghostsNear - 2*pelletDistance + 2000*maxScaredGhostsNear

# Abbreviation
better = betterEvaluationFunction
