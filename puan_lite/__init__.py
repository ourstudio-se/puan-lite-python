import numpy as np
import npycvx
import functools

from dataclasses import dataclass
from typing import List, Union, Iterator
from itertools import chain, starmap, repeat
from puan import variable
from puan.ndarray import ge_polyhedron

@dataclass
class ValuedVariable:

    id: str
    value: int

@dataclass
class GeLineq:

    valued_variables: List[ValuedVariable]
    bias: int

    def __hash__(self) -> int:
        return hash(
            tuple(
                sorted(
                    map(
                        lambda vv: (vv.id, vv.value),
                        self.valued_variables
                    )
                ) + [self.bias]
            )
        )

@dataclass
class Proposition:

    variables: List[str]

    def __hash__(self) -> int:
        return hash(
            tuple(
                sorted(self.variables) + [type(self)]
            )
        )

@dataclass
class AtMostOne(Proposition):

    def constraints(self) -> List[GeLineq]:
        return [
            GeLineq(
                valued_variables=list(
                    map(
                        lambda v: ValuedVariable(id=v, value=-1),
                        self.variables
                    )
                ),
                bias=1,
            )
        ]

@dataclass
class And(Proposition):

    def constraints(self) -> List[GeLineq]:
        return [
            GeLineq(
                valued_variables=list(
                    map(
                        lambda v: ValuedVariable(id=v, value=1),
                        self.variables
                    )
                ),
                bias=len(self.variables) * -1,
            )
        ]

@dataclass
class Or(Proposition):

    def constraints(self) -> List[GeLineq]:
        return [
            GeLineq(
                valued_variables=list(
                    map(
                        lambda v: ValuedVariable(id=v, value=1),
                        self.variables
                    )
                ),
                bias=-1,
            )
        ]

@dataclass
class Xor(Proposition):

    def constraints(self) -> List[GeLineq]:
        return [
            GeLineq(
                valued_variables=list(
                    map(
                        lambda v: ValuedVariable(id=v, value=1),
                        self.variables
                    )
                ),
                bias=-1,
            ),
            GeLineq(
                valued_variables=list(
                    map(
                        lambda v: ValuedVariable(id=v, value=-1),
                        self.variables
                    )
                ),
                bias=1,
            )
        ]

@dataclass
class XNor(Proposition):

    def constraints(self) -> List[GeLineq]:
        return list(
            map(
                lambda vo: GeLineq(
                    valued_variables=list(
                        chain(
                            [
                                ValuedVariable(
                                    id=vo,
                                    value=len(self.variables)-1,
                                )
                            ],
                            map(
                                lambda vi: ValuedVariable(
                                    id=vi,
                                    value=-1,
                                ),
                                set(self.variables) - set([vo]),
                            ),
                        )
                    ),
                    bias=0,
                ),
                self.variables,
            )
        )

@dataclass
class Nand(Proposition):

    def constraints(self) -> List[GeLineq]:
        return [
            GeLineq(
                valued_variables=list(
                    map(
                        lambda v: ValuedVariable(id=v, value=-1),
                        self.variables
                    )
                ),
                bias=len(self.variables)-1,
            )
        ]

@dataclass
class Nor(Proposition):

    def constraints(self) -> List[GeLineq]:
        return [
            GeLineq(
                valued_variables=list(
                    map(
                        lambda v: ValuedVariable(id=v, value=-1),
                        self.variables
                    )
                ),
                bias=0,
            )
        ]

@dataclass
class Impl:

    condition: Union[And, Or, Nand, Nor, Xor]
    consequence: Union[And, Or, Nand, Nor, Xor]

    def __hash__(self) -> int:
        return hash(
            self.condition.__hash__() + self.consequence.__hash__()
        )

    @property
    def variables(self) -> List[str]:
        return list(
            set(
                chain(
                    self.condition.variables,
                    self.consequence.variables,
                )
            )
        )

    def constraints(self) -> List[GeLineq]:

        if type(self.condition) == And and type(self.consequence) == And:
            return [
                GeLineq(
                    valued_variables=list(
                        chain(
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=-len(self.consequence.variables),
                                ),
                                self.condition.variables
                            ),
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=1,
                                ),
                                self.consequence.variables
                            )
                        )
                    ),
                    bias=(len(self.condition.variables)-1) * len(self.consequence.variables),
                )
            ]
        elif type(self.condition) == And and type(self.consequence) == Or:
            return [
                GeLineq(
                    valued_variables=list(
                        chain(
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=-1*len(self.consequence.variables),
                                ),
                                self.condition.variables
                            ),
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=1,
                                ),
                                self.consequence.variables
                            )
                        )
                    ),
                    bias=len(self.condition.variables)*len(self.consequence.variables)-1,
                )
            ]
        elif type(self.condition) == And and type(self.consequence) == Nand:
            return [
                GeLineq(
                    valued_variables=list(
                        chain(
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=-len(self.consequence.variables),
                                ),
                                self.condition.variables
                            ),
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=-1,
                                ),
                                self.consequence.variables
                            )
                        )
                    ),
                    bias=(len(self.consequence.variables)*len(self.condition.variables)+len(self.consequence.variables)-1),
                )
            ]
        elif type(self.condition) == And and type(self.consequence) == Nor:
            return [
                GeLineq(
                    valued_variables=list(
                        chain(
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=-1*len(self.consequence.variables),
                                ),
                                self.condition.variables
                            ),
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=-1,
                                ),
                                self.consequence.variables
                            )
                        )
                    ),
                    bias=len(self.consequence.variables)*len(self.condition.variables),
                )
            ]
        elif type(self.condition) == And and type(self.consequence) == AtMostOne:
            return [
                GeLineq(
                    valued_variables=list(
                        chain(
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=-len(self.consequence.variables),
                                ),
                                self.condition.variables
                            ),
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=-1,
                                ),
                                self.consequence.variables
                            )
                        ),
                    ),
                    bias=len(self.condition.variables)*len(self.consequence.variables)+1,
                )
            ]
        elif type(self.condition) == Or and type(self.consequence) == And:
            return list(
                map(
                    lambda vo: GeLineq(
                        valued_variables=list(
                            chain(
                                [
                                    ValuedVariable(
                                        id=vo,
                                        value=-len(self.consequence.variables),
                                    )
                                ],
                                map(
                                    lambda vi: ValuedVariable(
                                        id=vi,
                                        value=1,
                                    ),
                                    self.consequence.variables
                                )
                            )
                        ),
                        bias=0,
                    ),
                    self.condition.variables,
                )
            )
        elif type(self.condition) == Or and type(self.consequence) == Or:
            return [
                GeLineq(
                    valued_variables=list(
                        chain(
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=-1,
                                ),
                                self.condition.variables
                            ),
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=len(self.condition.variables),
                                ),
                                self.consequence.variables
                            )
                        )
                    ),
                    bias=0,
                )
            ]
        elif type(self.condition) == Or and type(self.consequence) == Nand:
            return list(
                map(
                    lambda vo: GeLineq(
                        valued_variables=list(
                            chain(
                                [
                                    ValuedVariable(
                                        id=vo,
                                        value=-1*len(self.consequence.variables),
                                    )
                                ],
                                map(
                                    lambda vi: ValuedVariable(
                                        id=vi,
                                        value=-1,
                                    ),
                                    self.consequence.variables,
                                )
                            )
                        ),
                        bias=len(self.consequence.variables) + len(self.condition.variables) - 1,
                    ),
                    self.condition.variables,
                )
            )
        elif type(self.condition) == Or and type(self.consequence) == Nor:
            return list(
                map(
                    lambda vo: GeLineq(
                        valued_variables=list(
                            chain(
                                [
                                    ValuedVariable(
                                        id=vo,
                                        value=-1*len(self.consequence.variables),
                                    )
                                ],
                                map(
                                    lambda vi: ValuedVariable(
                                        id=vi,
                                        value=-1,
                                    ),
                                    self.consequence.variables
                                )
                            )
                        ),
                        bias=len(self.consequence.variables),
                    ),
                    self.condition.variables,
                )
            )
        elif type(self.condition) == Or and type(self.consequence) == AtMostOne:
            return list(
                map(
                    lambda vo: GeLineq(
                        valued_variables=list(
                            chain(
                                [
                                    ValuedVariable(
                                        id=vo,
                                        value=-len(self.consequence.variables),
                                    )
                                ],
                                map(
                                    lambda vi: ValuedVariable(
                                        id=vi,
                                        value=-1,
                                    ),
                                    self.consequence.variables
                                )
                            )
                        ),
                        bias=len(self.consequence.variables) + 1,
                    ),
                    self.condition.variables,
                )
            )
        elif type(self.condition) == Nand and type(self.consequence) == And:
            return list(
                map(
                    lambda vo: GeLineq(
                        valued_variables=list(
                            chain(
                                [
                                    ValuedVariable(
                                        id=vo,
                                        value=len(self.consequence.variables),
                                    )
                                ],
                                map(
                                    lambda vi: ValuedVariable(
                                        id=vi,
                                        value=1,
                                    ),
                                    self.consequence.variables,
                                )
                            )
                        ),
                        bias=-len(self.consequence.variables),
                    ),
                    self.condition.variables,
                )
            )
        elif type(self.condition) == Nand and type(self.consequence) == Or:
            return [
                GeLineq(
                    valued_variables=list(
                        chain(
                            map(
                                lambda v: ValuedVariable(
                                    id=v,
                                    value=1,
                                ),
                                self.condition.variables,
                            ),
                            map(
                                lambda v: ValuedVariable(
                                    id=v,
                                    value=len(self.condition.variables),
                                ),
                                self.consequence.variables,
                            )
                        )
                    ),
                    bias=-len(self.condition.variables),
                ),
            ]
        elif type(self.condition) == Nand and type(self.consequence) == Nand:
            return list(
                map(
                    lambda vo: GeLineq(
                        valued_variables=list(
                            chain(
                                [
                                    ValuedVariable(
                                        id=vo,
                                        value=len(self.consequence.variables),
                                    )
                                ],
                                map(
                                    lambda vi: ValuedVariable(
                                        id=vi,
                                        value=-1,
                                    ),
                                    self.consequence.variables,
                                )
                            )
                        ),
                        bias=len(self.consequence.variables)-1,
                    ),
                    self.condition.variables,
                )
            )
        elif type(self.condition) == Nand and type(self.consequence) == Nor:
            return list(
                map(
                    lambda vo: GeLineq(
                        valued_variables=list(
                            chain(
                                [
                                    ValuedVariable(
                                        id=vo,
                                        value=len(self.consequence.variables),
                                    )
                                ],
                                map(
                                    lambda vi: ValuedVariable(
                                        id=vi,
                                        value=-1,
                                    ),
                                    self.consequence.variables,
                                )
                            )
                        ),
                        bias=0,
                    ),
                    self.condition.variables,
                )
            )
        elif type(self.condition) == Nor and type(self.consequence) == And:
            return [
                GeLineq(
                    valued_variables=list(
                        chain(
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=len(self.consequence.variables),
                                ),
                                self.condition.variables
                            ),
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=1,
                                ),
                                self.consequence.variables
                            )
                        )
                    ),
                    bias=-len(self.consequence.variables),
                )
            ]
        elif type(self.condition) == Nor and type(self.consequence) == Or:
            return [
                GeLineq(
                    valued_variables=list(
                        chain(
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=len(self.consequence.variables),
                                ),
                                self.condition.variables
                            ),
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=1,
                                ),
                                self.consequence.variables
                            )
                        )
                    ),
                    bias=-1,
                )
            ]
        elif type(self.condition) == Nor and type(self.consequence) == Nand:
            return [
                GeLineq(
                    valued_variables=list(
                        chain(
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=len(self.consequence.variables),
                                ),
                                self.condition.variables
                            ),
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=-1,
                                ),
                                self.consequence.variables
                            )
                        )
                    ),
                    bias=len(self.consequence.variables)-1,
                )
            ]
        elif type(self.condition) == Nor and type(self.consequence) == Nor:
            return [
                GeLineq(
                    valued_variables=list(
                        chain(
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=len(self.consequence.variables),
                                ),
                                self.condition.variables
                            ),
                            map(
                                lambda v: ValuedVariable(
                                    id=v, 
                                    value=-1,
                                ),
                                self.consequence.variables
                            )
                        )
                    ),
                    bias=0,
                )
            ]
        elif type(self.condition) == And and type(self.consequence) == Xor:
            return list(
                set(
                    chain(
                        Impl(
                            And(self.condition.variables),
                            Or(self.consequence.variables),
                        ).constraints(),
                        Impl(
                            And(self.condition.variables),
                            AtMostOne(self.consequence.variables),
                        ).constraints(),
                    )
                )
            )
        elif type(self.condition) == Or and type(self.consequence) == Xor:
            return list(
                set(
                    chain(
                        Impl(
                            Or(self.condition.variables),
                            Or(self.consequence.variables),
                        ).constraints(),
                        Impl(
                            Or(self.condition.variables),
                            AtMostOne(self.consequence.variables),
                        ).constraints(),
                    )
                )
            )
        else:
            raise Exception(
                f"Combination of {type(self.condition)} as condition and {type(self.consequence)} as consequence is not yet implemented."
            )

@dataclass
class Empt:

    variables: List[str]

    def constraints(self) -> List[GeLineq]:
        return [
            GeLineq(
                valued_variables=list(
                    map(
                        lambda v: ValuedVariable(id=v, value=0),
                        self.variables
                    )
                ),
                bias=0,
            )
        ]

@dataclass
class Conjunction:

    propositions: List[
        Union[
            And,
            Or,
            Nand,
            Nor,
            Impl,
            Empt,
            Xor,
        ]
    ]

    def variables(self) -> List[str]:
        return list(
            set(
                chain(
                    *map(
                        lambda p: p.variables,
                        self.propositions
                    )
                )
            )
        )

    def constraints(self) -> List[GeLineq]:
        return list(
            chain(
                *map(
                    lambda p: p.constraints(),
                    self.propositions
                )
            )
        )

    def to_ge_polyhedron(self) -> ge_polyhedron:
        variables = list(
            chain(
                ["#b"],
                sorted(self.variables())
            )
        )
        # Generate all constraints from each proposition
        constraints = self.constraints()
        
        # Initialize the matrix with zeros
        matrix = np.zeros((len(constraints), len(variables)))

        # Set the bias values in the first column
        matrix[:, 0] = -1 * np.array([c.bias for c in constraints])

        # Create a dictionary to map variable IDs to column indices for faster indexing
        variable_indices = dict(starmap(lambda i,v: (v,i), enumerate(variables)))

        # Create an array of corresponding values
        valued_variable_values = list(
            chain(
                *map(
                    lambda cn: map(
                        lambda vv: vv.value,
                        cn.valued_variables
                    ),
                    constraints,
                )
            )
        )

        # Use np.where to find the row and column indices where values should be assigned
        row_indices = list(
            chain(
                *starmap(
                    lambda i,cn: repeat(i, len(cn.valued_variables)),
                    enumerate(constraints),
                )
            )
        )
        column_indices = list(
            chain(
                *map(
                    lambda cn: map(
                        lambda vv: variable_indices[vv.id],
                        cn.valued_variables
                    ),
                    constraints,
                )
            )
        )

        # Use advanced indexing to assign values to the matrix
        matrix[row_indices, column_indices] = valued_variable_values

        return ge_polyhedron(
            matrix,
            variables=list(
                map(
                    lambda v: variable(id=v),
                    variables
                )
            ),
        )

    def solve(self, objectives: List[dict], minimize: bool = False) -> Iterator[dict]:

        # Convert system to a polyhedron
        polyhedron = self.to_ge_polyhedron()

        # Create partial function to solve the polyhedron
        # with multiple objectives
        solve_part_fn = functools.partial(
            npycvx.solve_lp, 
            *npycvx.convert_numpy(
                polyhedron.A,
                polyhedron.b,
            ), 
            minimize,
        )

        # Solve problems and return solutions as
        # dictionaries of variable IDs and values
        return starmap(
            lambda x,y: dict(
                zip(
                    x,
                    y,
                )
            ),
            zip(
                repeat(
                    map(
                        lambda x: x.id,
                        polyhedron.A.variables,
                    ),
                    len(objectives)
                ),
                map(
                    lambda x: x[1],
                    map(
                        solve_part_fn, 
                        map(
                            polyhedron.A.construct,
                            objectives
                        )
                    )
                )
            )
        )