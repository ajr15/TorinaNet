from abc import ABC, abstractclassmethod

class StopCondition (ABC):

    """Abastract stopping condition for the Iterate module"""

    @abstractclassmethod
    def check(self, rxn_graph, itr_count):
        """Method to check if stopping condition is met
        ARGS:
            - rxn_graph (RxnGraph): result reaction graph
            - itr_count (int): number of iterations made"""
        pass


    def check_msg(self, rxn_graph, itr_count):
        """Optional method to return a message when checking if condition is met."""
        return ""


class MaxIterNumber (StopCondition):

    """Stopping condition for stopping run after fixed amount of iterations"""

    def __init__(self, max_itr: int) -> None:
        self.max_itr = max_itr


    def check(self, rxn_graph, itr_count):
        if self.max_itr <= itr_count:
            return True
        else:
            return False

    
    def check_msg(self, rxn_graph, itr_count):
        msg = "{} out of {} iterations ran...\n"
        if self.max_itr < itr_count:
            msg += "STOPPING CONDITION IS MET, EXITING..."
        else:
            msg += "STOPPING CONDITION IS NOT MET, CONTINUEING..."
        return msg
