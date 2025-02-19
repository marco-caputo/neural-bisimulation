from typing import Optional


class SPA:
    def __init__(self, data: dict[str, dict[str, list[dict[str, float]]]]):
        set_ = set()
        self.data = data

        for state in self.data.keys():
            set_.add(state)
        self.states = sorted(list(set_))

        set_.clear()
        for state, value in self.data.items():
            if isinstance(value, dict):
                set_.update(value.keys())
        self.labels = list(set_)

    def pretty_print(self):
        for key, value in self.data.items():
            print(key, value)

    def squiggly_l(self, s, a):
        return self.data[s].get(a, [])


class DeterministicSPA:
    def __init__(self, data: dict[str, dict[str, dict[str, float]]]):
        set_ = set()
        self.data = data

        for state in self.data.keys():
            set_.add(state)
        self.states = sorted(list(set_))

        set_.clear()
        for state, value in self.data.items():
            if isinstance(value, dict):
                set_.update(value.keys())
        self.labels = list(set_)

    def pretty_print(self):
        for key, value in self.data.items():
            print(key, value)

    def index_of(self, s: str):
        """
        Returns the index corresponding to state s in this model.
        """
        return self.states.index(s)

    def actions(self, s: str | int) -> set[str]:
        """
        Returns the set of actions that can be executed in the state s.
        Equivalently, this method returns the set of actions that have assigned a probability distribution for
        the state s.
        """
        self._check_state(s)
        return set(self.data[s].keys())

    def get_probability(self, s: str | int, a: str, t: str | int):
        """
        Returns the probability of transitioning from state s to state t given the action a.
        If the state s has no outgoing transitions for the action a, an exception is raised.
        """
        s = self._check_state(s)
        t = self._check_state(t)
        self._check_action(a)
        if self.distribution(s, a) is None:
            raise ValueError(f'Action {a} not found in the state {s}.')

        return self.data[s].get(a, {}).get(t, 0)

    def distribution(self, s: str | int, a: str) -> Optional[dict[str, float]]:
        """
        Returns the distribution corresponding to the action a in the state s, if it exists.
        The distribution is a dictionary with keys being the target states and values being the probabilities
        of transitioning to that state. If the probability is 0, the target state is not included in the dictionary.

        :param s: The source state
        :param a: The action
        :return: The distribution of the action a in the state s
        """
        s = self._check_state(s)
        self._check_action(a)
        return self.data[s].get(a, None)

    def add_state(self, s: str):
        """
        Adds a new state to the model.
        If the state already exists, an exception is raised.

        :param s: The state to add
        """
        if s in self.states:
            raise ValueError(f'State {s} already exists.')
        self.states.append(s)
        self.data[s] = dict()

    def add_distribution(self, s: str | int, a: str, distribution: dict[str, float]):
        """
        Adds a new distribution to the given state-action pair.
        If the state-action pair already has a distribution, it is overwritten.
        Is not necessary for the action a to be already in the set of actions of DSPA.

        :param s: The source state
        :param a: The action
        :param distribution: The distribution of the action a in the state s
        """
        s = self._check_state(s)
        self._check_distribution(distribution)
        if a not in self.labels:
            self.labels.append(a)

        self.data[s][a] = dict(distribution)

    def _check_state(self, s: str | int) -> str:
        if isinstance(s, str) and s not in self.states:
            raise ValueError(f'Action {s} not found in the states.')
        if isinstance(s, int) and (s < 0 or s >= len(self.states)):
            raise ValueError(f'State {s} not found in the states.')
        if isinstance(s, int):
            s = self.states[s]
        return s

    def _check_action(self, a: str):
        if a not in self.labels:
            raise ValueError(f'Action {a} not found in the labels.')

    def _check_distribution(self, distribution: dict[str, float]):
        for p in distribution.values():
            if p < 0 or 1 < p:
                raise ValueError('The probabilities must be in the range [0, 1].')
        if abs(sum(distribution.values()) - 1) > 1e-6:
            raise ValueError('The sum of probabilities must be equal to 1.')

    def __iter__(self):
        return iter(self.states)

    def __repr__(self):
        return f'DSPA({self.data})'

