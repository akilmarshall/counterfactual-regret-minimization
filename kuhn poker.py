from random import random, shuffle
import numpy as np
from typing import List


class KuhnTrainer:
    """Kuhn Poker and CFR problem definitions. """
    PASS = 0
    BET = 1
    NUM_ACTIONS = 2
# str -> Node
    nodeMap = dict()

    class Node:
        """Information set node class definition. """

        def __init__(self):
            """Kuhn node definitions. """
            self.infoSet: str = ''
            self.NUM_ACTIONS = KuhnTrainer().NUM_ACTIONS
            self.regretSum = np.zeros(self.NUM_ACTIONS, dtype=float)
            self.strategy = np.zeros(self.NUM_ACTIONS, dtype=float)
            self.strategySum = np.zeros(self.NUM_ACTIONS, dtype=float)

        def getStrategy(self, realizationWeight: float):
            """Get current information set mixed strategy through regret-matching. """
            normalizingSum: float = 0
            for a in range(self.NUM_ACTIONS):
                self.strategy[a] = self.regretSum[a] if self.regretSum[a] > 0 else 0
                normalizingSum += self.strategy[a]
            for a in range(self.NUM_ACTIONS):
                if normalizingSum > 0:
                    self.strategy[a] /= normalizingSum
                else:
                    self.strategy[a] = 1 / self.NUM_ACTIONS
                self.strategySum[a] += realizationWeight * self.strategy[a]
            return self.strategy

        def getAverageStrategy(self):
            """Get average information set mixed strategy across all training iterations. """
            avgStrategy = np.zeros(self.NUM_ACTIONS, dtype=float)
            normalizingSum: float = sum(self.strategySum)
            for a in range(self.NUM_ACTIONS):
                if normalizingSum > 0:
                    avgStrategy[a] = self.strategySum[a] / normalizingSum
                else:
                    avgStrategy[a] = 1 / self.NUM_ACTIONS
            return avgStrategy

        def __str__(self):
            """Get information set string representation. """
            return f'{self.infoSet}: {self.getAverageStrategy()}'

    def __init__(self):
        pass

    def train(self, iterations: int) -> None:
        """Train Kuhn poker. """
        cards: List[int] = [1, 2, 3]
        util: float = 0
        for i in range(iterations):
            """Shuffle cards. """
            shuffle(cards)
            util += self.cfr(cards, '', 1, 1)
        print(f'Average game value: {util / iterations}')
        for n in self.nodeMap.values():
            print(n)

    def cfr(self, cards: List[int], history: str, p0: float, p1: float) -> float:
        """Counterfactual regret minimization iteration. """
        plays: int = len(history)
        player: int = plays % 2
        opponent = 1 - player

        """Return payoff for terminal states. """
        if plays > 1:
            terminalPass: bool = history[plays - 1] == 'p'
            doubleBet: bool = history[-2:] == 'bb'
            isPlayerCardHigher: bool = cards[player] > cards[opponent]
            if terminalPass:
                if history == 'pp':
                    return 1 if isPlayerCardHigher else -1
                else:
                    return 1
            elif doubleBet:
                return 2 if isPlayerCardHigher else -2

        infoSet: str = f'{cards[player]} {history}'

        """Get information set node or create it if nonexistant. """
        node: Node = self.nodeMap.get(infoSet)
        if node is None:
            node = self.Node()
            node.infoSet = infoSet
            self.nodeMap[infoSet] = node

        """For each action, recursively call cfr with additional history and probability. """
        strategy: float = node.getStrategy(p0 if player == 0 else p1)
        util: float = np.zeros(self.NUM_ACTIONS, dtype=float)
        nodeUtil: float = 0
        for a in range(self.NUM_ACTIONS):
            nextHistory = history + ('p' if a == 0 else 'b')
            if player == 0:
                util[a] = -self.cfr(cards, nextHistory, p0 * strategy[a], p1)
            else:
                util[a] = -self.cfr(cards, nextHistory, p0, p1 * strategy[a])

            nodeUtil += strategy[a] * util[a]

        """For each action, compute and accumulate counterfactual regret. """
        for a in range(self.NUM_ACTIONS):
            regret: float = util[a] - nodeUtil
            node.regretSum[a] += (p1 if player == 0 else p0) * regret

        return nodeUtil


def main():
    iterations: int = 1000000
    trainer: KuhnTrainer = KuhnTrainer()
    trainer.train(iterations)


if __name__ == '__main__':
    main()
