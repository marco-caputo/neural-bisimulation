from ApproxBisimulation import PFSP
from ApproxBisimulation import V1_PFSP_approximate_bisimulation
from typing import Set, Dict, List
from werkzeug.datastructures import MultiDict

import random

class Main:

    v1=0
    v2=0
    v3=0

    def start(self):
        states = set(["S", "a1", "a2", "b1", "b2","b3","c1","c2","F1","F2","F3","F4"])

        pfspA = PFSP.ProbabilisticFiniteStateProcess(states, "S")

        pfspB = PFSP.ProbabilisticFiniteStateProcess(states, "S")

        pfspA.add_transition("S", "action", "a1", 0.5)
        pfspA.add_transition("S", "action", "a2", 0.5)
        pfspA.add_transition("a1", "action", "b1", 0.3)
        pfspA.add_transition("a1", "action", "b2", 0.7)
        pfspA.add_transition("a2", "action", "b2", 0.5)
        pfspA.add_transition("a2", "action", "b3", 0.5)
        pfspA.add_transition("b1", "action", "c1", 1)
        pfspA.add_transition("b2", "action", "c1", 0.5)
        pfspA.add_transition("b2", "action", "c2", 0.5)
        pfspA.add_transition("b3", "action", "c2", 1)
        pfspA.add_transition("c1", "action", "F1", 1)
        pfspA.add_transition("c2", "action", "F2", 1)

        
        pfspB.add_transition("S", "action", "a1", 0.5)
        pfspB.add_transition("S", "action", "a2", 0.5)
        self.refresh2RandomValues()
        pfspB.add_transition("a1", "action", "b1", self.v1)
        pfspB.add_transition("a1", "action", "b2", self.v2)
        self.refresh2RandomValues()
        pfspB.add_transition("a2", "action", "b2", self.v1)
        pfspB.add_transition("a2", "action", "b3", self.v2)
        pfspB.add_transition("b1", "action", "c1", 1)
        self.refresh2RandomValues()
        pfspB.add_transition("b2", "action", "c1", self.v1)
        pfspB.add_transition("b2", "action", "c2", self.v2)
        pfspB.add_transition("b3", "action", "c2", 1)
        self.refresh3RandomValues()
        pfspB.add_transition("c1", "action", "F1", self.v1)
        pfspB.add_transition("c1", "action", "F2", self.v2)
        pfspB.add_transition("c1", "action", "F3", self.v3)
        self.refresh3RandomValues()
        pfspB.add_transition("c2", "action", "F1", self.v1)
        pfspB.add_transition("c2", "action", "F2", self.v2)
        pfspB.add_transition("c2", "action", "F3", self.v3)

        # pfspB.add_transition("S", "action", "a1", 0.5)
        # pfspB.add_transition("S", "action", "a2", 0.5)
        # pfspB.add_transition("a1", "action", "F1", 0.5)
        # pfspB.add_transition("a1", "action", "F2", 0.5)
        # pfspB.add_transition("a2", "action", "F2", 0.5)
        # pfspB.add_transition("a2", "action", "F3", 0.5)



        # approxBis = V1_PFSP_approximate_bisimulation.ApproxBisV1PFSPManager(pfspA, pfspA)
        approxBis = V1_PFSP_approximate_bisimulation.ApproxBisV1PFSPManager(pfspA, pfspB)

        print("\nThe matrix:\n" + str(approxBis._search_matrix) + "\n")

        approxBis.evaluate_probabilistic_approximate_bisimulation(0.3, True)


    def refresh3RandomValues(self):
        for i in range(random.randrange(10)*7):
            random.randrange(10)
        self.v1 = random.randrange(100)/100
        self.v2 = random.randrange(100)/100
        self.v3 = 1 - (self.v1 + self.v2)

    def refresh2RandomValues(self):
        for i in range(random.randrange(10)*7):
            random.randrange(10)
        self.v1 = random.randrange(100)/100
        self.v2 = 1 - self.v1

Main().start()