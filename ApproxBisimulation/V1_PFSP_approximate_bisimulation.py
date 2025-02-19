from ApproxBisimulation import PFSP
from typing import Set, Dict, List

#TODO remove BisPy from the library and from .gitmodules

class ApproxBisV1PFSPManager:
    """Approximate bisimulation manager for PFSP with only one action and no tau moves (|V|=1)"""

    studied_structures: tuple[ PFSP.ProbabilisticFiniteStateProcess, PFSP.ProbabilisticFiniteStateProcess] = tuple()
    _search_matrix: List[ List[Dict[str, Dict[str, float]]] ] = list()
    """
    This is a couple of matrix organizing the nodes for a more efficient research.\n
    :List[Dict[str, Dict[str, float]]]: is a PFSP, which is represented as a set of layers.\n
    :Dict[str, Dict[str, float]]: is a dict of states with their respective couples (final_state, prob).\n
    :Dict[str, float]: is a couple of final state targets with the respective cumulative probability to be reached.
    """


    def __init__(self, pfspA: PFSP.ProbabilisticFiniteStateProcess, pfspB: PFSP.ProbabilisticFiniteStateProcess):
        if pfspA.all_observable_actions().__len__() != 1 | pfspA.all_observable_actions().__len__() != 1:
            raise ValueError("Only single-action PFSP are accepted (1 observable action excluding tau actions).")

        self.studied_structures = (pfspA, pfspB)
        self._search_matrix = list([list(), list()]) #layer lists representing the two pfsp are initialized here...
        self.calculate_all_nodes_cumulative_probabilities() #...while the other nestled elements are initialized and defined here.


    def evaluate_probabilistic_approximate_bisimulation(self, epsilon: float) -> float:
        """
        Evaluates if the two FSP are approximate bisimilar under an 'epsilon' tolerance.\n
        The function works with an adaptation fo the Hausdorff distance for probabilistic FSP.\n
        :return float: A bisimilitude percentage between the two transition systems.
        """
        ...
    

    def _prob_approx_bis_evaluation(self, layer_ID: int, epsilon: float) -> float:
        """
        Iterative component of the function 'evaluate_probabilistic_approximate_bisimulation'.\n
        From the starting couple of layers from the two pfsp, this function calculates
        the Hausdorff difference is calculated between each couple of layers until both of them reach their final layer.\n
        :return float: A bisimilitude percentage between the two transition systems.
        """
        ...

        
    def calculate_all_nodes_cumulative_probabilities(self) -> None:
        """
        For each state of the two pfsp a dict of reachable final states and the cumulative probability
        is calculated and the result added into the 'research_matrix'.\n
        Different actions are considered as different paths leading to equivalent states. So as 'eventually-merging deviations'.
        """
        for i in [0,1]:
            self._nodes_cumulative_probs_calculation(i, 0, self.studied_structures[i].start)


    def _nodes_cumulative_probs_calculation(self, pfsp_ID: int, layer_ID: int, state: str) -> Dict[str, float]:
        """
        Recursive component of the function 'calculate_all_nodes_cumulative_probabilities'.\n
        Ths function is recursive (Depth-First Search) exploring every possible paths from the given layer of states, 
        but at the same time it avoids repeating the same operations by consulting the matrix.\n
        :param pfsp_ID: First ID used to navigate through the matrix
        :param layer_ID: Second ID used to navigate through the matrix
        :param state: The state to analyse and add into the matrix.
        """
        if self._search_matrix[pfsp_ID].__len__() == layer_ID: #check the need to add a layer
            self._search_matrix[pfsp_ID].append(dict()) #create a new representative layer in the matrix

        if state in self._search_matrix[pfsp_ID][layer_ID]: #check if the calculation has been done already
            return self._search_matrix[pfsp_ID][layer_ID].get(state)
        
        selected_pfsp = self.studied_structures[pfsp_ID]
        action = selected_pfsp.all_observable_actions().pop() #this set is supposed to have just one element

        #create a new representative state in the matrix and make a reference available
        self._search_matrix[pfsp_ID][layer_ID].update({state: dict()})
        matrix_state_cumulativeProbs_to_finalStates = self._search_matrix[pfsp_ID][layer_ID].get(state)

        transitions = selected_pfsp.target_states(state, action)
        if transitions.__len__() == 0: #check if it's a final state
            matrix_state_cumulativeProbs_to_finalStates.update({state: 1}) #adds itself as target with 100% probability
        else:
            for target_prob in transitions: #doing the deep search for every transition
                for target in target_prob:
                    prob_to_reach_target = target_prob.get(target)
                    target_cumulativeProbs_to_finalStates = self._nodes_cumulative_probs_calculation(pfsp_ID, layer_ID+1, target)

                    for final_state_to_reach in target_cumulativeProbs_to_finalStates: #compute the probabilistic derivation tree for every possible final state
                        probability_to_sum = target_cumulativeProbs_to_finalStates.get(final_state_to_reach) * prob_to_reach_target
                        old_probability = matrix_state_cumulativeProbs_to_finalStates.get(final_state_to_reach) or 0.0
                        new_probability = round(probability_to_sum + old_probability, 3)

                        matrix_state_cumulativeProbs_to_finalStates.update({final_state_to_reach: new_probability})

        return matrix_state_cumulativeProbs_to_finalStates