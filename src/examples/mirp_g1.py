"""
SM Harwood
18 June 2020

An example inspired by MIRPLib Group 1 instance
https://mirplib.scl.gatech.edu/sites/default/files/LR1_2_DR1_3_VC2_V6a.txt
Some modifications
"""
from formulations import MIRP

def get_mirp(time_horizon):
    """
    Define a specific problem given a time horizon
    """
    # Create a routing problem
    # specify cargo size/vessel capacity and time horizon
    cargo_size = 300
    mirp = MIRP(cargo_size, time_horizon)

    # Define demand node data
    # Inventory is Initial inventory at start of time horizon
    # (see add_nodes)
    d_names =     ['D1', 'D2', 'D3']
    inventories = [221,  215,  175]
    rates =       [-34,  -31,  -25]
    tankages =    [374,  403,  300]
    fees =        [60,   82,   94]
    demand_port_fees = dict(zip(d_names, fees))

    # Add demand nodes to problem
    for (name, inv, rate, tank) in zip(d_names, inventories, rates, tankages):
        mirp.add_nodes(name, inv, rate, tank)

    # Define supply node data
    s_names =     ['S1', 'S2']
    inventories = [220,  270]
    rates =       [47,   42]
    tankages =    [376,  420]
    fees =        [30,   85]
    supply_port_fees = dict(zip(s_names, fees))

    # Add supply nodes to problem
    for (name, inv, rate, tank) in zip(s_names, inventories, rates, tankages):
        mirp.add_nodes(name, inv, rate, tank)

    # Arcs
    # km/day
    vessel_speed = 665.0
    # dollars per km
    cost_per_unit_distance = 0.09
    # Distances (km) between (S1, S2, D1, D2, D3) x (S1, S2, D1, D2, D3)
    distance_matrix = [[0.00,       212.34,     5305.34,    5484.21,    5459.31],
                       [212.34,     0.00,       5496.06,    5674.36,    5655.55],
                       [5305.34,    5496.06,    0.00,       181.69,     380.30],
                       [5484.21,    5674.36,    181.69,     0.00,       386.66],
                       [5459.31,    5655.55,    380.30,     386.66,     0.00]]
    combined_ports = s_names + d_names
    def distance_function(port1, port2):
        i = combined_ports.index(port1)
        j = combined_ports.index(port2)
        return distance_matrix[i][j]

    mirp.add_travel_arcs(
        distance_function,
        vessel_speed,
        cost_per_unit_distance,
        supply_port_fees,
        demand_port_fees
    )
    mirp.add_exit_arcs()
    mirp.add_entry_arcs(time_limit=14)
    return mirp
