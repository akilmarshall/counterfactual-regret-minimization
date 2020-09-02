from pettingzoo.classic import rps_v0
from enum import Enum, auto
from random import randint, random
from typing import List
import numpy as np


class Action(Enum):
    rock = auto()
    paper = auto()
    scissors = auto()


class RPSTrainer:
    """Learn to play rock-paper-scissors using regret matching. """

    def __init__(self):
        # definitions
        self.ROCK = 0
        self.PAPER = 1
        self.SCISSORS = 2
        self.NUM_ACTIONS = 3
        self.regretSum = np.zeros(self.NUM_ACTIONS, dtype=float)
        self.oppRegretSum = np.zeros(self.NUM_ACTIONS, dtype=float)
        self.strategy = np.zeros(self.NUM_ACTIONS, dtype=float)
        self.strategySum = np.zeros(self.NUM_ACTIONS, dtype=float)
        self.oppStrategySum = np.zeros(self.NUM_ACTIONS, dtype=float)
        self.oppStrategy = np.zeros(self.NUM_ACTIONS, dtype=float)

    def getStrategy(self):
        """Get current mixed strategy through regret-matching. """
        # used when only a single player uses regret matching
        normalizingSum = 0
        for a in range(self.NUM_ACTIONS):
            self.strategy[a] = self.regretSum[a] if self.regretSum[a] > 0 else 0
            normalizingSum += self.strategy[a]
        for a in range(self.NUM_ACTIONS):
            if normalizingSum > 0:
                self.strategy[a] /= normalizingSum
            else:
                self.strategy[a] = 1 / self.NUM_ACTIONS
            self.strategySum[a] += self.strategy[a]
        return self.strategy

    def computeStrategies(self):
        "Compute regret matched strategies for both players. returns a list tuple."
        # used when both players use regret matching
        normalizingSum = [0, 0]
        for a in range(self.NUM_ACTIONS):
            self.strategy[a] = self.regretSum[a] if self.regretSum[a] > 0 else 0
            self.oppStrategy[a] = self.oppRegretSum[a] if self.oppRegretSum[a] > 0 else 0
            normalizingSum[0] += self.strategy[a]
            normalizingSum[1] += self.oppStrategy[a]
        for a in range(self.NUM_ACTIONS):
            if normalizingSum[0] > 0:
                self.strategy[a] /= normalizingSum[0]
            else:
                self.strategy[a] = 1 / self.NUM_ACTIONS
            self.strategySum[a] += self.strategy[a]

            if normalizingSum[1] > 0:
                self.oppStrategy[a] /= normalizingSum[1]
            else:
                self.oppStrategy[a] = 1 / self.NUM_ACTIONS
            self.oppStrategySum[a] += self.oppStrategy[a]
        return (self.strategy, self.oppStrategy)

    def getAction(self, strategy):
        r: float = random()
        a: int = 0
        cumulativeProbability: float = 0
        while a < self.NUM_ACTIONS - 1:
            cumulativeProbability += strategy[a]
            if r < cumulativeProbability:
                break
            a += 1
        return a

    def train(self, n: int):
        """Run the training routine for n iterations. """
        actionUtility = np.zeros(self.NUM_ACTIONS, dtype=float)
        for _ in range(n):
            # Get regret-matched mixed-strategy actions
            strategy, oppStrategy = self.computeStrategies()
            myAction = self.getAction(strategy)
            otherAction = self.getAction(self.oppStrategy)
            # Compute action utilities
            actionUtility[otherAction] = 0
            actionUtility[0 if otherAction ==
                          self.NUM_ACTIONS - 1 else otherAction + 1] = 1
            actionUtility[self.NUM_ACTIONS -
                          1 if otherAction == 0 else otherAction - 1] = -1
            # Accumulate action regrets
            for a in range(self.NUM_ACTIONS):
                self.regretSum[a] += actionUtility[a] - actionUtility[myAction]
                self.oppRegretSum[a] += -(actionUtility[a] +
                                          actionUtility[otherAction])

    def getAverageStrategy(self):
        avgStrategy = [np.zeros(self.NUM_ACTIONS), np.zeros(self.NUM_ACTIONS)]
        normalizingSum = (sum(self.strategySum), sum(self.oppStrategySum))
        for a in range(self.NUM_ACTIONS):
            if normalizingSum[0] > 0:
                avgStrategy[0][a] = self.strategySum[a] / normalizingSum[0]
            else:
                avgStrategy[0][a] = 1 / self.NUM_ACTIONS

            if normalizingSum[1] > 0:
                avgStrategy[1][a] = self.oppStrategySum[a] / normalizingSum[1]
            else:
                avgStrategy[1][a] = 1 / self.NUM_ACTIONS

        return tuple(avgStrategy)


def main():
    trainer = RPSTrainer()
    trainer.train(1000000)
    print(trainer.getAverageStrategy())


if __name__ == '__main__':
    main()
