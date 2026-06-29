
from domain.compounds import compounds
from domain.dissociation import Dissociation
from numpy.typing import NDArray
from scipy.optimize import least_squares, OptimizeResult
import numpy as np

MASS_BALANCE_WEIGHT: float = 1000.0
EQUILIBRIUM_WEIGHT: float = 1.0
ENERGY_WEIGHT: float = 1.0
DISSOCIATING_COMPOUNDS: tuple[str, ...] = (
    "Carbon_Dioxide",
    "Carbon_Monoxide",
    "Hydrogen_Monatomic",
    "Hydroxyl",
    "Methane",
    "Oxygen_Monatomic",
    "Water",
)


i: dict[str, int] = {
    "Carbon": 0,
    "Carbon_Dioxide": 1,
    "Carbon_Monoxide": 2,
    "Hydrogen": 3,
    "Hydrogen_Monatomic": 4,
    "Hydroxyl": 5,
    "Methane": 6,
    "Oxygen": 7,
    "Oxygen_Monatomic": 8,
    "Water": 9,
    "T": 10
}

r: dict[str, int] = {
    "H_mass_balance": 0,
    "C_mass_balance": 1,
    "O_mass_balance": 2,
    "Carbon_Dioxide": 3,
    "Carbon_Monoxide": 4,
    "Hydrogen_Monatomic": 5,
    "Hydroxyl": 6,
    "Methane": 7,
    "Oxygen_Monatomic": 8,
    "Water": 9,
    "Energy_balance": 10
}

init_log_guess: NDArray[np.float64] = np.array([-50, -0.47756, -50, -50, -50, -50, -50, -3, -50, -0.17653, 3000])

init_atoms: dict[int, float] = {1: 1.332, 6: 0.333, 8: 1.334}

initial_enthalpy: float = -24.932709

DISSOCIATION_OBJECTS: dict[str, Dissociation] = {
    compound: Dissociation(compound, set(compounds[compound].composition.keys()))
    for compound in DISSOCIATING_COMPOUNDS
}


def mass_balance_residuals(log_guess: NDArray[np.float64]) -> NDArray[np.float64]:

    mbr = np.zeros_like(log_guess)
    guess_H = 2*(10**log_guess[i["Hydrogen"]]) + 10**log_guess[i["Hydrogen_Monatomic"]] + 10**log_guess[i["Hydroxyl"]] + 4*(10**log_guess[i["Methane"]]) + 2*(10**log_guess[i["Water"]])
    mbr[r["H_mass_balance"]] = init_atoms[1] - guess_H
    guess_C = 10**log_guess[i["Carbon"]] + 10**log_guess[i["Carbon_Dioxide"]] + 10**log_guess[i["Carbon_Monoxide"]] + 10**log_guess[i["Methane"]]
    mbr[r["C_mass_balance"]] = init_atoms[6] - guess_C
    guess_O = 2*(10**log_guess[i["Carbon_Dioxide"]]) + 10**log_guess[i["Carbon_Monoxide"]] + 10**log_guess[i["Hydroxyl"]] + 2*(10**log_guess[i["Oxygen"]]) + 10**log_guess[i["Oxygen_Monatomic"]] + 10**log_guess[i["Water"]]
    mbr[r["O_mass_balance"]] = init_atoms[8] - guess_O
    return mbr # Consider putting this in log-space


def equil_residuals(log_guess: NDArray[np.float64]) -> NDArray[np.float64]:

    ebr = np.zeros_like(log_guess)
    for compound, diss_obj in DISSOCIATION_OBJECTS.items():
        resid = diss_obj.equilibrium_residual(log_guess, i, 1)
        ebr[r[compound]] = resid
    return ebr


def energy_residual(log_guess: NDArray[np.float64]) -> float:

    product_enthalpy: float = 0
    temp = log_guess[i["T"]]
    for compound_id, indx in i.items():
        if compound_id == "T":
            continue
        compound = compounds[compound_id]
        product_enthalpy += 10**log_guess[indx] * (compound.SH(temp) + compound.stdHf)
    return product_enthalpy - initial_enthalpy


def residual_function(log_guess: NDArray[np.float64]) -> NDArray[np.float64]:

    residuals = np.zeros_like(log_guess, dtype=np.float64)
    residuals += MASS_BALANCE_WEIGHT * mass_balance_residuals(log_guess)
    residuals += EQUILIBRIUM_WEIGHT * equil_residuals(log_guess)
    residuals[r["Energy_balance"]] = ENERGY_WEIGHT * energy_residual(log_guess)
    return residuals


def equilibrate(init_log_guess: NDArray[np.float64]):
    result: OptimizeResult = least_squares(
        residual_function,
        init_log_guess,
        max_nfev=10000
    )
    return result


print(equilibrate(init_log_guess))