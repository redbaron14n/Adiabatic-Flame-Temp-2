# ###################
# Ian Janes
# Professor Don Lipkin
# Adiabatic Flame Temperature
# Dissociation Reaction Class File
# ###################

from domain.compounds import compounds
from math import isclose, log10
from numpy.typing import NDArray
import numpy as np


class Dissociation:

    def __init__(self, m_id: str):

        self._set_molecule(m_id)
        self._set_comp()
        self._set_nonsolids()


    ########################################
    # Getters and Setters
    ########################################


    @property
    def molecule_id(self) -> str:

        """
        :return: The ID of the molecule that is dissociating in this reaction.
        """

        return self._molecule


    def _set_molecule(self, id: str):

        self._molecule: str = id


    @property
    def composition(self) -> dict[str, float]:

        return self._comp
    

    def _set_comp(self):

        self._comp = compounds[self._molecule].composition


    def _set_nonsolids(self):

        self._nonsolids: set[str] = {self._molecule}
        for species in self._comp.keys():
            compound = compounds[species]
            if compound.state != "s":
                self._nonsolids.add(compound.id)


    ########################################
    # Private Methods
    ########################################


    def _validate_guess(self, log_guess: NDArray[np.float64], item_indices: dict[str, int]):

        if len(log_guess) != len(item_indices):
            raise ValueError(
                f"Guess list length does not match number of items.\n"
                f"Guess: {log_guess}\n"
                f"Item Indices: {item_indices}"
            )
        elif not all(species in item_indices for species in ({self._molecule} | self._comp.keys())):
            raise ValueError(
                f"Guess list does not contain all species in the reaction.\n"
                f"Guess: {log_guess}\n"
                f"Item Indices: {item_indices}\n"
                f"Reaction Species: {set(self._molecule) | self._comp.keys()}"
            )


    def _calc_pressure_exp(self) -> float:

        """
        Calculates and returns the pressure exponent for the equilibrium residual calculation based on the stoichiometry of the reaction.
        """

        # stoich_dict = self._stoich
        # mfrm = compounds[self._molecule].formula
        # rfrms = {compounds[r].formula for r in (self._radicals & self._nonsolids)}
        # return stoich_dict[mfrm] - sum(stoich_dict[r] for r in rfrms)
        exp = 1.
        for radical in (self._comp.keys() & self._nonsolids): # Carbon activity excluded
            exp -= self._comp[radical]
        return exp
    

    # def _calc_gas_moles(self, log_guess: NDArray[np.float64], species_indices: dict[str, int]) -> float:

    #     total_moles = 0.0
    #     for species, indx in species_indices.items():
    #         if (species != "T") and (compounds[species].state != "s"):
    #             total_moles += 10**log_guess[indx]
    #     return total_moles
        

    def _calc_log_conc_product(self, log_guess: NDArray[np.float64], species_indices: dict[str, int]) -> float:

        product = log_guess[species_indices[self._molecule]]
        for species, coeff in self._comp.items(): # Carbon activity included
            product -= log_guess[species_indices[species]] * coeff
        return product
    

    def _calc_log_pres_factor(self, pressure_fraction: float) -> float:

        exponent = self._calc_pressure_exp()
        if isclose(exponent, 0.0):
            return 0.0
        return exponent * log10(pressure_fraction)


    ########################################
    # Public Methods
    ########################################
    

    def get_log_eq_constant(self, temperature: float) -> float:

        compound = compounds[self._molecule]
        log_eq_constant = compound.logKf(temperature)
        return log_eq_constant


    def equilibrium_residual(self, log_guess: NDArray[np.float64], species_indices: dict[str, int], pressure_fraction: float) -> float:

        self._validate_guess(log_guess, species_indices)
        temp = log_guess[species_indices["T"]]
        log_conc_product = self._calc_log_conc_product(log_guess, species_indices)
        log_pressure_factor = self._calc_log_pres_factor(pressure_fraction)
        log_ecc = self.get_log_eq_constant(temp)
        return log_conc_product + log_pressure_factor - log_ecc