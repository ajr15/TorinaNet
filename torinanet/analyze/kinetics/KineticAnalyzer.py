# package for kinetic analysis of RxnGraph objects
# shows the relative concentration of each specie at different times
# calculates the rates of the reactions with time
# finds concentrations at steady state (if exist) or maximal rates / concentrations
import numpy as np
from scipy.integrate import ode
from typing import List, Optional
from ...core.RxnGraph import BaseRxnGraph
from ...core import Reaction, Specie

class KineticAnalyzer:

    """General kinetic analyzer for reaction graphs"""

    def __init__(self, rxn_graph: BaseRxnGraph, rate_constant_property: str="k"):
        self.rxn_graph = rxn_graph
        self.rate_constant_property = rate_constant_property
        # initializing specie dictionary
        self._specie_d = self._build_specie_idx_dict(rxn_graph)
        # initializing solver results
        self._concs = None
        self._ts = None

    def get_rate(self, rxn: Reaction, step: Optional[int]=None) -> float:
        """Method to get the rate of a reaction at a given simulation step"""
        return rxn.properties["k"] * np.product([self.get_concentration(s, step) for s in rxn.reactants])

    def get_max_rate(self, rxn: Reaction) -> float:
        """Get the maximal reaction rate"""
        max_rate = 0
        for step in range(self.get_n_steps()):
            r = self.get_rate(rxn, step)
            if r > max_rate:
                max_rate = r
        return max_rate

    def get_concentration(self, specie: Specie, step: Optional[int]=None) -> float:
        """Get specie concentration at a simulation step"""
        if self._concs:
            # securing case with too large step number
            if self.rxn_graph.has_specie(specie):
                if step is None:
                    # return concentration in last step is step is not defined
                    sid = self.rxn_graph.make_unique_id(specie)
                    idx = self._specie_d[sid]
                    return self._concs[-1][idx]
                if step >= len(self._concs):
                    print("WARNING: supplied step is larger than the total number of steps, returning concentration on last step")
                    step = len(self._concs) - 1
                # returning the concentration
                sid = self.rxn_graph.make_unique_id(specie)
                idx = self._specie_d[sid]
                return self._concs[step][idx]
            else:
                raise ValueError("Supplied specie is not in reaction graph")
        else:
            raise RuntimeError("No simulation was ran! before reading concentrations you must call the solve_kinetics method at least once")

    @staticmethod
    def _build_specie_idx_dict(rxn_graph: BaseRxnGraph) -> dict:
        """Get dictionary of specie ID -> index in vector. used internally for solver"""
        d = {}
        for i, s in enumerate(rxn_graph.species):
            sid = rxn_graph.make_unique_id(s)
            d[sid] = i
        return d

    def _build_f(self):
        """build the function for the solver"""
        # list of functions (conc_vec -> conc_derr) for each reaction in the graph
        rfuncs = []
        ks = []
        rids = []
        pids = []
        for reaction in self.rxn_graph.reactions:
            rids.append([self._specie_d[self.rxn_graph.make_unique_id(s)] for s in reaction.reactants])
            pids.append([self._specie_d[self.rxn_graph.make_unique_id(s)] for s in reaction.products])
            ks.append(reaction.properties[self.rate_constant_property])
            rate = lambda t, concs, k, ridxs: k * np.product([concs[i] for i in ridxs])
            # defining the reaction function
            def rfunc(t, concs, k, ridxs, pidxs):
                diff = np.zeros(len(concs))
                r = rate(t, concs, k, ridxs)
                for i in ridxs:
                    diff[i] -= r
                for i in pidxs:
                    diff[i] += r
                return diff
            # appending r_func to list
            rfuncs.append(rfunc)
        # returning the total function
        return lambda t, concs: np.sum([rfunc(t, concs, k, ridxs, pidxs) for rfunc, k, ridxs, pidxs in zip(rfuncs, ks, rids, pids)], axis=0)

    def solve_kinetics(self, simulation_time: float, timestep: float, initial_concs: List[float], **solver_kwargs):
        """Solve the rate equations at given conditions.
        ARGS:
            - simulation_time (float): total simulation time
            - timestep (float): time of each simulation step
            - initial_concs (List[float]): list of initial specie concentrations
            - **solver_kwargs: keywords for scipy.integrate.ode.set_integrator method
        RETURNS:
            None"""
        # building target function
        target_f = self._build_f()
        # setting up solver
        solver = ode(target_f)
        solver.set_integrator(**solver_kwargs)
        solver.set_initial_value(y=initial_concs)
        # solving the ODE
        t = 0
        self._concs = [initial_concs]
        self._ts = [0]
        while solver.successful() and t < simulation_time:
            t = t + timestep
            self._ts.append(t)
            self._concs.append(solver.integrate(t))

    def find_max_reaction_rates(self, simulation_time: float, timestep: float, initial_concs: List[float], **solver_kwargs):
        """Solve the rate equations at given conditions and follow only the maximal reaction rates.
        ARGS:
            - simulation_time (float): total simulation time
            - timestep (float): time of each simulation step
            - initial_concs (List[float]): list of initial specie concentrations
            - **solver_kwargs: keywords for scipy.integrate.ode.set_integrator method
        RETURNS:
            None"""
        # building target function
        target_f = self._build_f()
        # setting up solver
        solver = ode(target_f)
        solver.set_integrator(**solver_kwargs)
        solver.set_initial_value(y=initial_concs)
        # solving the ODE
        t = 0
        self._concs = [initial_concs]
        self._ts = [0]
        max_rates = {self.rxn_graph.make_unique_id(rxn): self.get_rate(rxn) for rxn in self.rxn_graph.reactions}
        while solver.successful() and t < simulation_time:
            t = t + timestep
            self._ts[0] = t
            self._concs[0] = solver.integrate(t)
            print("time =", t)
            for rxn in self.rxn_graph.reactions:
                rid = self.rxn_graph.make_unique_id(rxn)
                rate = self.get_rate(rxn)
                if rate > max_rates[rid]:
                    max_rates[rid] = rate
        print("DONE !")
        return max_rates

    def get_n_steps(self) -> int:
        """Get the total number of simulation steps. If no simulation was ran, returns None"""
        if self._concs:
            return len(self._concs)

    def get_specie_index(self, sid: str):
        if sid in self._specie_d:
            return self._specie_d[sid]
        else:
            raise ValueError("Specie is not modeled in this analyzer")
