from typing import Set, Dict

from AQTSMetrics import SPA


class FiniteStateProcess:
    TAU = "τ"

    def __init__(self, states: Set[str], start_state: str, transition_function: Dict[str, Dict[str, Set[str]]] = None):
        """
        Initializes a Finite State Process (FSP).

        :param states: Finite set of states (K)
        :param start_state: Initial state (p0) from K
        :param transition_function: A dictionary representing Δ (K × V → 2^K), where
                                    keys are (state, action) pairs mapping to a set of resulting states.
        """
        if start_state not in states:
            raise ValueError("Start state must be in the set of states K")

        self.states = set(states)
        self.start = start_state
        self.transition_function = dict(transition_function) if transition_function is not None else dict()

    def target_states(self, state: str, action: str) -> Set[str]:
        """
        Returns the set of possible next states given a state and an action.
        If the given state is not in the set of states, a ValueError is raised.
        """
        if state not in self.states:
            raise ValueError("State must be in the set of states")
        return self.transition_function.get(state, {}).get(action, set())

    def extension(self) -> Dict[str, Set[str]]:
        """
        Returns the extension function of the FSP in form of a dictionary.
        The extension function is a mapping from states to the set of actions allowed in that state.
        """
        return {state: set(self.transition_function.get(state, {}).keys()) for state in self.states}

    def all_actions(self) -> Set[str]:
        """
        Returns the set of all actions allowed in the FSP, including the unobservable action τ.
        """
        return {action for state in self.states for action in self.actions(state)} | {self.TAU}

    def actions(self, state: str) -> Set[str]:
        """
        Returns the set of actions allowed in a given state.
        If the given state is not in the set of states, a ValueError is raised.
        """
        if state not in self.states:
            raise ValueError("State must be in the set of states")

        return set(self.transition_function.get(state, {}).keys())

    def is_tau(self, action: str) -> bool:
        """
        Returns True if the given action is the unobservable action τ.
        """
        return action == self.TAU

    def add_state(self, state: str):
        """
        Adds a state to the set of states.
        If the given state is already in the set of states, nothing happens.
        """
        self.states.add(state)

    def add_transition(self, state: str, action: str, target_state: str):
        """
        Adds a transition to the transition function.
        If the given states are not in the set of states, a ValueError is raised.
        """
        if state not in self.states or target_state not in self.states:
            raise ValueError("States must be in the set of states")

        if state not in self.transition_function:
            self.transition_function[state] = {}
        if action not in self.transition_function[state]:
            self.transition_function[state][action] = set()

        self.transition_function[state][action].add(target_state)

    def to_spa(self) -> SPA:
        """
        Converts the FSP to a SPA model.
        The conversion is made maintaining the same set of states and transitions, so every state in the resulting
        SPA model has the same set of actions as in the FSP model.
        Moreover, for each action one probability distribution is defined for the SPA state, where each possible
        target state has the same probability of occurring.
        """

        data = dict()
        for state in self.states:
            data[state] = dict()
            for action in self.actions(state):
                prob = 1. / len(self.target_states(state, action))
                data[state][action] = [{target: prob for target in self.target_states(state, action)}]

        return SPA(data)

    def __repr__(self):
        return f"FiniteStateProcess(K={self.states}, p0={self.start}, Δ={self.transition_function}, E={self.extension()})"
