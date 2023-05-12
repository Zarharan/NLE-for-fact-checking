from abc import ABC, abstractmethod


class NLIStructure(ABC):
    @abstractmethod
    def predict_nli(self, premise, hypothesis):
        ''' The implementation of this function should return the NLI label ID with "entailment": 0, "neutral": 1, and "contradiction": 2.

        :param premise: The premise
        :type premise: str
        :param hypothesis: The hypothesis
        :type hypothesis: str
        
        :returns: The NLI label ID ("entailment": 0, "neutral": 1, and "contradiction": 2)
        :rtype: int
        '''        
        pass