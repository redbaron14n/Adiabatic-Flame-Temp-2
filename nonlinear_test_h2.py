from domain.compounds import compounds
from domain.dissociation import Dissociation
from numpy.typing import NDArray
from scipy.optimize import least_squares
import numpy as np


i: dict[str, int] = {
    "Hydrogen": 0,
    "Hydrogen_Monatomic": 1,
    "Hydroxyl": 2,
    "Nitric_Oxide": 3,
    "Nitrogen": 4,
    "Nitrogen_Dioxide": 5,
    "Nitrogen_Monatomic": 6,
    "Oxygen": 7,
    "Oxygen_Monatomic": 8,
    "Water": 9,
    "T": 10
}

r: dict[str, int] = {
    "H_mass_balance": 0,
    "N_mass_balance": 1,
    "O_mass_balance": 2,
    "Hydrogen_Monatomic": 3,
    "Hydroxyl": 4,
    "Nitric_Oxide": 5,
    "Nitrogen_Dioxide": 6,
    "Nitrogen_Monatomic": 7,
    "Oxygen_Monatomic": 8,
    "Water": 9,
    "Energy_balance": 10
}

init_log_guess: NDArray[np.float64] = np.array([-3., -50., -50., -50., 0.12450, -50., -50., -50., -50., -0.17653, 3000.])

init_atoms: dict[int, float] = {1: 1.334, 7: 2.664, 8: 0.666}

initial_enthalpy: float = 0


def mass_balance_residuals(log_guess: NDArray[np.float64]) -> NDArray[np.float64]:

    mbr = np.zeros_like(log_guess)

    guess_H = 2*(10**log_guess[i["Hydrogen"]]) + 10**log_guess[i["Hydrogen_Monatomic"]] + 10**log_guess[i["Hydroxyl"]] + 2*(10**log_guess[i["Water"]])
    mbr[r["H_mass_balance"]] = init_atoms[1] - guess_H

    guess_N = 10**log_guess[i["Nitric_Oxide"]] + 2*(10**log_guess[i["Nitrogen"]]) + 10**log_guess[i["Nitrogen_Dioxide"]] + 10**log_guess[i["Nitrogen_Monatomic"]]
    mbr[r["N_mass_balance"]] = init_atoms[7] - guess_N

    guess_O = 10**log_guess[i["Hydroxyl"]] + 2*(10**log_guess[i["Oxygen"]]) + 10**log_guess[i["Oxygen_Monatomic"]] + 10**log_guess[i["Water"]]
    mbr[r["O_mass_balance"]] = init_atoms[8] - guess_O

    return mbr # Consider putting this in log-space


def calc_pressure_fraction(log_guess: NDArray[np.float64], pressure: float) -> float:

    gas_moles = 0.
    for comp, indx in i.items():
        if comp == "T":
            continue
        elif compounds[comp].state == "s":
            continue
        gas_moles += 10**log_guess[indx]
    return pressure / gas_moles


def equil_residuals(log_guess: NDArray[np.float64], pressure: float) -> NDArray[np.float64]:

    ebr = np.zeros_like(log_guess)
    frac = calc_pressure_fraction(log_guess, pressure)
    for compound in ["Hydrogen_Monatomic", "Hydroxyl", "Nitric_Oxide", "Nitrogen_Dioxide", "Nitrogen_Monatomic", "Oxygen_Monatomic", "Water"]:
        diss_obj = Dissociation(compound)
        resid = diss_obj.equilibrium_residual(log_guess, i, frac)
        ebr[r[compound]] = resid
    return ebr


def energy_residual(log_guess: NDArray[np.float64]) -> float:

    product_enthalpy: float = 0
    temp = log_guess[i["T"]]
    for compound_id, indx in i.items():
        if (compound_id == "T") or (compound_id == "Carbon"):
            continue
        compound = compounds[compound_id]
        product_enthalpy += 10**log_guess[indx] * (compound.SH(temp) + compound.stdHf)
    return product_enthalpy - initial_enthalpy


def residual_function(log_guess: NDArray[np.float64], pressure: float) -> NDArray[np.float64]:

    residuals = np.zeros_like(log_guess)
    residuals += mass_balance_residuals(log_guess)
    residuals += equil_residuals(log_guess, pressure)
    residuals[r["Energy_balance"]] = energy_residual(log_guess)
    return residuals


def equilibrate(init_log_guess: NDArray[np.float64], pressure: float):
    result = least_squares(residual_function, init_log_guess, args=(pressure,))
    return result


print(equilibrate(init_log_guess, 1.))