from typing import Sequence

import networkx as nx

from data.mol import Atom
from data.knot import Knot


def get_knotgraph(_atoms: Sequence[Atom], _molgraph: nx.graph) -> nx.graph:
    """
    Generate a graph of rings (Knot Objects).

    in:
    _atoms: A list of Atoms with their xyz coordinates in Angstroms.
    _molgraph: Molecular graph.

    out:
    graph: generated graph of the knots.

    """
    knots = get_knots(_atoms, _molgraph)
    edges = get_knots_connectivity(knots)
    graph = nx.Graph(edges) # generate mathematical graph as networkx Graph object

    return graph


def get_knots(_atoms: Sequence[Atom], _molgraph: nx.graph) -> Sequence[Knot]:
    """
    Function that gets the geometric center of each ring of the molecule and initializes the Knot Objects for each monocycle.

    in:
    _atoms: A list of Atoms with their xyz coordinates in Angstroms.
    _cycles: A list of monocycles. Each monocycle is a list of atom indices.

    out:
    knots: A list of Knots (= monocycles).

    """
    cycles = nx.minimum_cycle_basis(_molgraph)
    knots = [] # initialize list to return
    i = 0
    for cycle in cycles:
        cycle_atoms = ''
        x_knot = y_knot = z_knot = 0
        for atom in cycle:
            cycle_atoms += _atoms[atom].element
            x_knot += _atoms[atom].x
            y_knot += _atoms[atom].y
            z_knot += _atoms[atom].z
        
        knot_type = 'bn' # when dealing with different types of rings we will implement a function here to deal with the ring\
                         # type identificatioin
        _knot = Knot(i, knot_type, x_knot/len(cycle), y_knot/len(cycle),
                     z_knot/len(cycle), [_atoms[x] for x in cycle])
        i += 1
        knots.append(_knot)
    
    return knots


def get_knots_connectivity(_knots: Sequence[Knot]) -> Sequence[tuple]:
    """
    get_connectivity(_knots: list(Knot)) -> list(tuple)

    Find out which Knot objects are connected and return those connections as a list of tuples ( = edges).

    in:
    _knots: A list of Knot objects.

    out:
    edges: A list of tuples ( = edges) which represent which Knot objects are connected.

    """
    edges = []
    for i in range(len(_knots)):
        for j in range(i + 1, len(_knots)):
            i_atoms = set(_knots[i].atoms)
            j_atoms = set(_knots[j].atoms)
            if i_atoms & j_atoms:
                edges.append((i,j))

    return edges

