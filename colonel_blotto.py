from random import random
import numpy as np
from itertools import product
from random import choice


class ColonelBlottoTrainer:

    def __init__(self):
        self.pure_strategies = [x for x in self._generate_pure_strategies()]
        self.num_actions: int = len(self.pure_strategies)
        # the players board states intialized randomly
        self.colonel_blotto = choice(self.pure_strategies)
        self.boba_fett = choice(self.pure_strategies)

        self.regret_sum = np.zeros(self.num_actions, dtype=float)
        self.strategy = np.zeros(self.num_actions, dtype=float)
        self.strategy_sum = np.zeros(self.num_actions, dtype=float)

    def _generate_pure_strategies(self, n: int = 4, s: int = 17):
        """Returns a generator for the following set {(i_1, ..., i_n,) : i_1, ..., i_n \in \Z_{+},i_1 + ... + i_n = s}. """
        for x in product(range(s + 1), repeat=n):
            if sum(x) == s:
                yield x

    def _calculate_utility(self, strategy_a, strategy_b):
        """Calculate the utility of strategy_a against strategy_b. """
        score = 0
        for a, b in zip(strategy_a, strategy_b):
            if a > b:
                score += 1
            elif a < b:
                score -= 1
        return score

    def get_utility(self) -> int:
        return self._calculate_utility(self.colonel_blotto, self.boba_fett)

    def compute_strategy(self):
        """Compute the strategy. """
        normalizing_sum = 0
        for a in range(self.num_actions):
            self.strategy[a] = self.regret_sum[a] if self.regret_sum[a] > 0 else 0
            normalizing_sum += self.strategy[a]

        for a in range(self.num_actions):
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                self.strategy[a] = 1 / self.num_actions
            self.strategy_sum[a] += self.strategy[a]

    def get_action(self):
        r: float = random()
        a: int = 0
        cumulative_probability: float = 0
        while a < self.num_actions - 1:
            cumulative_probability += self.strategy[a]
            if r < cumulative_probability:
                break
            a += 1
        return self.pure_strategies[a]

    def train(self, n):
        action_utility = np.zeros(self.num_actions, dtype=float)
        for _ in range(n):
            # Train colonel_blotto
            self.compute_strategy()
            action = self.get_action()

            # Accumulate action regrets
            for a in range(self.num_actions):
                action = self.pure_strategies[a]
                self.regret_sum[a] += self._calculate_utility(
                    action, self.boba_fett) - self.get_utility()

            # pick new board states to calculate more regrets
            self.colonel_blotto = choice(self.pure_strategies)
            self.boba_fett = choice(self.pure_strategies)

    def get_average_strategy(self):
        avg_strategy = np.zeros(self.num_actions, dtype=float)
        normalizing_sum = sum(self.strategy_sum)

        for a in range(self.num_actions):
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 1 / self.num_actions

        return avg_strategy


def main():
    trainer = ColonelBlottoTrainer()
    trainer.train(10000)
    avg = trainer.get_average_strategy()
    print(max(avg))
    print(trainer.pure_strategies[avg.argmax()])


if __name__ == '__main__':
    main()
