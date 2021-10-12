# Ref -> https://www.youtube.com/watch?v=ELE2_Mftqoc&t=6416s&ab_channel=freeCodeCamp.org

# Import libraries
import numpy as np
import matplotlib as plt


class GridWorld(object):
    def __init__(self, m, n, magicSquares):
        self.grid = np.zeros((m, n))
        self.m = m  # row
        self.n = n  # col
        self.stateSpace = [i for i in range(self.m * self.n)]
        self.stateSpace.remove(self.m * selfn - 1)
        self.stateSpacePlus = [i for i in range(self.m * self.n)]
        self.actionSpace = {'U': -self.m, 'D': self.m, 'L': -1, 'R': 1}
        self.possibleActions = ['U', 'D', 'L', 'R']
        self.addMagicSquares(magicSquares)
        self.agentPosition = 0

    def addMagicSquares(self, magicSquares):
        self.magicSquares = magicSquares
        i = 2
        for square in magicSquares:
            x = square // self.m  # floor
            y = square % self.n
            self.grid[x][y] = i
            i += 1
            x = magicSquares[square] // self.n
            y = magicSquares[square] % self.n
            self.grid[x][y] = i
            i += 1

    def isTerminalState(self, state):
        return state in self.stateSpacePlus and state not in self.stateSpace

    def getAgentRowAndColumn(self):
        x = self.agentPosition // self.n
        y = self.agentPosition % self.n

    def setState(self, state):
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 0
        self.agentPosition = state
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 1

    def offGridMove(self, newState, oldState):
        if newState not in self.stateSpacePlus:
            return True
        elif oldState % self.m == 0 and newState % self.m == self.m - 1:
            return True
        elif oldState % self.m == self.m - 1 and newState % self.m == 0:
            return True
        else:
            return False

    def step(self, action):
        x, y = self.getAgentRowAndColumn()
        resultingState = self.agentPosition + self.actionSpace[action]

        if resultingState in self.magicSquares.keys():
            resultingState = self.magicSquares[resultingState]

        reward = -1 if not self.isTerminalState(resultingState) else 0
        if not self.offGridMove(resultingState, self.agentPosition):
            self.setState(resultingState)

            return resultingState, reward, \
                   self.isTerminalState(self.agentPosition), None

        else:
            return self.agentPosition, reward, \
                   self.isTerminalState(self.agentPosition), None

    def reset(self):
        self.agentPosition = 0
        self.grid = np.zeros((self.m, self.n))
        self.addMagicSquares(self.magicSquares)
        return self.agentPosition

    def render(self):
        print('----------------------------------------')
        for row in self.grid:
            for col in row:
                if col == 0:
                    print('-', end='\t')
                elif col == 1:
                    print('X', end='\t')
                elif col == 2:
                    print('Ain', end='\t')
                elif col == 3:
                    print('Aout', end='\t')
                elif col == 4:
                    print('Bin', end='\t')
                else:
                    print('Bout', end='\t')
            print('\n')
        print('----------------------------------------')
