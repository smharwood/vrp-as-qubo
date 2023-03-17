"""
SM Harwood
31 January 2023

Random MIRP generator
"""
from typing import Optional, Union
from numbers import Real
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import uniform
from ..tools.sampling import WrapperSampler, Sampleable, Sampleable_Type, RVT
from ..applications import MIRP

class RandomMIRP:
    """
    A class to construct random instances of a Maritime Inventory Routing Problem
    """
    def __init__(self,
            cargo_size: Union[Real,Sampleable],
            time_horizon: Union[Real,Sampleable],
            num_supply_ports: Union[int,Sampleable],
            num_demand_ports: Union[int,Sampleable],
            inventory_init_supply: Union[ArrayLike,Sampleable],
            inventory_init_demand: Union[ArrayLike,Sampleable],
            inventory_rate_supply: Union[ArrayLike,Sampleable],
            inventory_rate_demand: Union[ArrayLike,Sampleable],
            inventory_cap_supply:  Union[ArrayLike,Sampleable],
            inventory_cap_demand:  Union[ArrayLike,Sampleable],
            travel_times: Union[ArrayLike,Sampleable],
            travel_cost_per_unit_time: Optional[Union[Real,Sampleable]]=None,
            supply_port_fees: Optional[Union[ArrayLike,Sampleable]]=None,
            demand_port_fees: Optional[Union[ArrayLike,Sampleable]]=None,
            seed: Optional[int]=None
        ):
        self.cargo_size = cargo_size
        self.time_horizon = time_horizon
        self.num_supply_ports = num_supply_ports
        self.num_demand_ports = num_demand_ports
        if np.isscalar(inventory_init_supply):
            inventory_init_supply = inventory_init_supply*np.ones(1)
        if np.isscalar(inventory_init_demand):
            inventory_init_demand = inventory_init_demand*np.ones(1)
        if np.isscalar(inventory_rate_supply):
            inventory_rate_supply = inventory_rate_supply*np.ones(1)
        if np.isscalar(inventory_rate_demand):
            inventory_rate_demand = inventory_rate_demand*np.ones(1)
        if np.isscalar(inventory_cap_supply):
            inventory_cap_supply = inventory_cap_supply*np.ones(1)
        if np.isscalar(inventory_cap_demand):
            inventory_cap_demand = inventory_cap_demand*np.ones(1)
        self.inventory_init_supply = inventory_init_supply
        self.inventory_init_demand = inventory_init_demand
        self.inventory_rate_supply = inventory_rate_supply
        self.inventory_rate_demand = inventory_rate_demand
        self.inventory_cap_supply = inventory_cap_supply
        self.inventory_cap_demand = inventory_cap_demand
        self.travel_times = travel_times
        self.travel_cost_per_unit_time = travel_cost_per_unit_time
        self.supply_port_fees = supply_port_fees
        self.demand_port_fees = demand_port_fees
        self.seed = seed

    def get_random_mirp(self, size: int=1):
        """
        Get random instance(s) of the MIRP
        """
        # set random seed
        np.random.seed(self.seed)

        mirps = []
        for _ in range(size):
            # For any input that is a distribution/random variable, sample it
            if isinstance(self.time_horizon, Sampleable_Type):
                time_horizon = self.time_horizon.rvs()
            else:
                time_horizon = self.time_horizon

            if isinstance(self.cargo_size, Sampleable_Type):
                cargo_size = self.cargo_size.rvs()
            else:
                cargo_size = self.cargo_size

            if isinstance(self.num_supply_ports, Sampleable_Type):
                num_supply_ports = self.num_supply_ports.rvs()
            else:
                num_supply_ports = self.num_supply_ports

            if isinstance(self.num_demand_ports, Sampleable_Type):
                num_demand_ports = self.num_demand_ports.rvs()
            else:
                num_demand_ports = self.num_demand_ports

            # Supply
            if isinstance(self.inventory_init_supply, Sampleable_Type):
                inventory_init_supply = self.inventory_init_supply.rvs(size=num_supply_ports)
            else:
                inventory_init_supply = np.asarray(self.inventory_init_supply)
            assert (inventory_init_supply > 0).all(), "Supply initial inventory should be positive"
            assert len(inventory_init_supply) == num_supply_ports, \
                "Length of inventory_init_supply is incorrect"

            if isinstance(self.inventory_rate_supply, Sampleable_Type):
                inventory_rate_supply = self.inventory_rate_supply.rvs(size=num_supply_ports)
            else:
                inventory_rate_supply = np.asarray(self.inventory_rate_supply)
            assert (inventory_rate_supply > 0).all(), "Supply inventory rate should be positive"
            assert len(inventory_rate_supply) == num_supply_ports, \
                "Length of inventory_rate_supply is incorrect"

            if isinstance(self.inventory_cap_supply, Sampleable_Type):
                inventory_cap_supply = self.inventory_cap_supply.rvs(size=num_supply_ports)
            else:
                inventory_cap_supply = np.asarray(self.inventory_cap_supply)
            assert (inventory_cap_supply > 0).all(), "Supply inventory capacity should be positive"
            assert len(inventory_cap_supply) == num_supply_ports, \
                "Length of inventory_cap_supply is incorrect"

            # Demand
            if isinstance(self.inventory_init_demand, Sampleable_Type):
                inventory_init_demand = self.inventory_init_demand.rvs(size=num_demand_ports)
            else:
                inventory_init_demand = np.asarray(self.inventory_init_demand)
            assert (inventory_init_demand > 0).all(), "demand initial inventory should be positive"
            assert len(inventory_init_demand) == num_demand_ports, \
                "Length of inventory_init_demand is incorrect"

            if isinstance(self.inventory_rate_demand, Sampleable_Type):
                inventory_rate_demand = self.inventory_rate_demand.rvs(size=num_demand_ports)
            else:
                inventory_rate_demand = np.asarray(self.inventory_rate_demand)
            assert (inventory_rate_demand < 0).all(), "demand inventory rate should be negative"
            assert len(inventory_rate_demand) == num_demand_ports, \
                "Length of inventory_rate_demand is incorrect"

            if isinstance(self.inventory_cap_demand, Sampleable_Type):
                inventory_cap_demand = self.inventory_cap_demand.rvs(size=num_demand_ports)
            else:
                inventory_cap_demand = np.asarray(self.inventory_cap_demand)
            assert (inventory_cap_demand > 0).all(), "demand inventory capacity should be positive"
            assert len(inventory_cap_demand) == num_demand_ports, \
                "Length of inventory_cap_demand is incorrect"

            # Travel arcs
            n_total = num_supply_ports + num_demand_ports
            if isinstance(self.travel_times, Sampleable_Type):
                travel_times = self.travel_times.rvs(size=(n_total,n_total))
                # zero out diagonal and make symmetric
                # set lower triangle equal to upper triangle
                for i in range(n_total):
                    travel_times[i,i] = 0
                    for j in range(i+1,n_total):
                        travel_times[j,i] = travel_times[i,j]
            else:
                travel_times = np.asarray(self.travel_times)
            assert (travel_times >= 0).all(), "Travel times should be non-negative"
            assert travel_times.shape == (n_total,n_total),\
                "Shape of travel_times is incorrect"

            if self.travel_cost_per_unit_time is None:
                # travel time will be proxy for cost
                travel_cost_per_unit_time = 1.0
            elif isinstance(self.travel_cost_per_unit_time, Sampleable_Type):
                travel_cost_per_unit_time = self.travel_cost_per_unit_time.rvs()
            else:
                travel_cost_per_unit_time = self.travel_cost_per_unit_time
            assert travel_cost_per_unit_time >= 0,\
                "Travel cost per unit time should be non-negative"

            if self.supply_port_fees is None:
                supply_port_fees = np.zeros(num_supply_ports)
            elif isinstance(self.supply_port_fees, Sampleable_Type):
                supply_port_fees = self.supply_port_fees.rvs(size=num_supply_ports)
            else:
                supply_port_fees = self.supply_port_fees
            assert len(supply_port_fees) == num_supply_ports,\
                "Length of supply_port_fees is incorrect"

            if self.demand_port_fees is None:
                demand_port_fees = np.zeros(num_demand_ports)
            elif isinstance(self.demand_port_fees, Sampleable_Type):
                demand_port_fees = self.demand_port_fees.rvs(size=num_demand_ports)
            else:
                demand_port_fees = self.demand_port_fees
            assert len(demand_port_fees) == num_demand_ports,\
                "Length of demand_port_fees is incorrect"

            # Construct MIRP object
            mirp = MIRP(cargo_size, time_horizon)

            # Add nodes
            # Supply
            sp_fees = {}
            for i, (init, rate, cap) in enumerate(
                zip(inventory_init_supply, inventory_rate_supply, inventory_cap_supply)
            ):
                name = f"S{i+1}"
                mirp.add_nodes(name, init, rate, cap)
                sp_fees[name] = supply_port_fees[i]
            # Demand
            dp_fees = {}
            for i, (init, rate, cap) in enumerate(
                zip(inventory_init_demand, inventory_rate_demand, inventory_cap_demand)
            ):
                name = f"D{i+1}"
                mirp.add_nodes(name, init, rate, cap)
                dp_fees[name] = demand_port_fees[i]

            # Add arcs
            # "Distance function" will just give travel times, and we will use
            # a vessel speed of 1
            # This function gets used immediately then we don't care anymore
            # so theres no late binding issue
            combined_ports = mirp.supply_ports + mirp.demand_ports
            def distance_function(port1, port2):
                i = combined_ports.index(port1)
                j = combined_ports.index(port2)
                return travel_times[i][j]
            mirp.add_travel_arcs(
                distance_function,
                vessel_speed=1,
                cost_per_unit_distance=travel_cost_per_unit_time,
                supply_port_fees=sp_fees,
                demand_port_fees=dp_fees
            )
            mirp.add_exit_arcs()
            # How many vessels should we have?
            # ultimately, the problem will get modified when the specific
            # formulations are constructed.
            # But, some initial entry arcs are probably a good idea...
            mirp.add_entry_arcs(time_limit=time_horizon/10)
            mirps.append(mirp)
        # end for loop

        if size == 1:
            return mirps[0]
        return mirps


def get_generator(
        num_supply_ports: int,
        num_demand_ports: int,
        time_horizon: Real,
        travel_times: Optional[Sampleable]=None,
        time_windows: Optional[Sampleable]=None,
        time_between_windows: Optional[Sampleable]=None
    ) -> RandomMIRP:
    """
    RandomMIRP is maybe TOO random to be useful
    This method will try to give some flexibility while setting some other
    values to sensible defaults.
    """
    # uniform(loc, scale) has support [loc, loc+scale]
    # These are sort of arbitrary distributions...
    # maybe representative of "maritime" problems from e.g. MIRPLib
    # If time between windows is small, then potentially a lot of ships are needed
    if travel_times is None:
        travel_times = uniform(loc=10, scale=10)
    if time_windows is None:
        time_windows = uniform(loc=2, scale=2)
    if time_between_windows is None:
        time_between_windows = uniform(loc=10, scale=20)

    if isinstance(travel_times, RVT):
        travel_times = WrapperSampler(travel_times)
    if isinstance(time_windows, RVT):
        time_windows = WrapperSampler(time_windows)
    if isinstance(time_between_windows, RVT):
        time_between_windows = WrapperSampler(time_between_windows)

    # The random MIRP is defined by the distributions of:
    #   time window (widths),
    #   time between time windows, and
    #   travel times
    # Looking at how time windows are defined
    # (see ..applications.mirp.py)
    # we get
    # time window width = (cap - size)/|rate|
    # time between time windows = size/|rate|
    # ==>
    # rate = size/(time between time windows)
    # cap = (time window width)*|rate| + size
    #     = size*(time window width)/(time between time windows) + size
    #     = size*((time window width)/(time between time windows) + 1)
    # Thus, rate and capacity consistent with the desired time window statistics
    #  do not depend on cargo size (just the ratios) - pick an arbitrary value
    cargo_size = 1.0

    # Supply and demand nodes might be better modeled with different distributions,
    # but at least these will give different values when sampled
    inventory_rate_supply =  cargo_size/time_between_windows
    inventory_rate_demand = -cargo_size/time_between_windows
    inventory_cap_supply =   time_windows*inventory_rate_supply  + cargo_size
    inventory_cap_demand = -(time_windows*inventory_rate_demand) + cargo_size
    # To make sure the first time windows are positive,
    # we should follows some constraints:
    # The first time window for supply is the earliest a ship can load a full load;
    #   init < cargo_size
    # The first time window for demand is the earliest a ship can discharge/unload;
    #   init > cap - size
    inventory_init_supply = uniform(loc=0, scale=cargo_size)
    inventory_init_demand = inventory_cap_demand - cargo_size/2

    mirp_gen = RandomMIRP(
        cargo_size,
        time_horizon,
        num_supply_ports,
        num_demand_ports,
        inventory_init_supply,
        inventory_init_demand,
        inventory_rate_supply,
        inventory_rate_demand,
        inventory_cap_supply,
        inventory_cap_demand,
        travel_times
    )
    return mirp_gen
