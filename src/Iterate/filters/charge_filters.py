from abc import ABC, abstractclassmethod
from ...core.Specie import Specie

class ChargeFilter (ABC):

    """Abstract filter for filtering species based on charge. Used for charge iterations"""

    @abstractclassmethod
    def check(self, specie: Specie) -> bool:
        """Method to check if a specie stands will get filtered or not"""
        pass


class MaxAbsCharge (ChargeFilter):

    def __init__(self, max_abs_charge: int) -> None:
        self.max_abs_charge = max_abs_charge

    def check(self, specie: Specie) -> bool:
        return abs(specie.charge) <= self.max_abs_charge