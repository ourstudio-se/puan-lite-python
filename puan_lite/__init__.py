import numpy as np
import npycvx
import functools

from dataclasses import dataclass
from typing import List, Union, Iterator, Optional
from itertools import chain, starmap, repeat
from puan import variable
from puan.ndarray import ge_polyhedron

from sympy.logic.boolalg import to_cnf
from sympy.parsing.sympy_parser import parse_expr

def impl_death_rows(impl_prop) -> list:

    """
        Converts an implication proposition into a list of constraints, naively.
        NOTE: This leads into an exponential number of constraints in relation to the number of variables.
    """
    def string_to_constraint(string: str) -> Optional[GeLineq]:
        if string[0] == "(" and string[-1] == ")":
            return string_to_constraint(string[1:-1])

        literals = set(string.split(" | "))
        atoms = list(set(string.replace("~","").split(" | ")))
        lit_join = "".join(literals)
        if any(map(lambda a: lit_join.count(a) > 1, atoms)):
            # If true here, expression is tautology
            return None
            
        return GeLineq(
            valued_variables=list(
                map(
                    lambda v: ValuedVariable(
                        id=v[1:] if v[0] == "~" else v,
                        value=-1 if v[0] == "~" else 1,
                    ),
                    literals,
                ),
            ),
            bias=-1 + string.count("~"),
        )

    return list(
        set(
            filter(
                lambda x: x is not None,
                map(
                    string_to_constraint,
                    str(
                        to_cnf(
                            parse_expr(
                                impl_prop.to_string(),
                            )
                        )
                    ).split(" & ")
                )
            )
        )
    )

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

class Proposition:

    def __init__(self, *variables: List[Union["Proposition", str]]):
        # Make the list unique after initialization
        object.__setattr__(self, "variables", list(set(variables)))

    def __hash__(self) -> int:
        return hash(
            tuple(
                sorted(
                    chain(
                        map(
                            lambda v: hash(v),
                            self.variables
                        ),
                        [hash(type(self))]
                    )
                )
            )
        )

    def atoms(self) -> List[str]:
        return list(
            set(
                chain(
                    filter(
                        lambda x: type(x) == str,
                        self.variables,
                    ),
                    *map(
                        lambda x: x.atoms(),
                        filter(
                            lambda x: type(x) != str,
                            self.variables,
                        )
                    )
                )
            )
        )

    def is_complex(self) -> bool:
        return any(map(lambda x: type(x) != str, self.variables))

    def variable_strings(self) -> List[str]:
        return list(
            map(
                lambda x: x if type(x) == str else x.to_string(),
                self.variables,
            )
        )

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

    def to_string(self) -> str:
        return f"({' + '.join(self.variable_strings())}) <= 1"

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

    def to_string(self) -> str:
        return ' & '.join(self.variable_strings())

    def constraints(self) -> List[GeLineq]:
        return list(
            chain(
                *map(
                    lambda p: p.constraints(),
                    self.variables,
                )
            )
        )

    def to_ge_polyhedron(self) -> ge_polyhedron:
        variables = list(
            chain(
                ["#b"],
                sorted(self.atoms())
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

    def to_string(self) -> str:
        return ' | '.join(self.variable_strings())

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

    def to_string(self) -> str:
        return ' ^ '.join(self.variable_strings())

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

    def to_string(self) -> str:
        return f"~({' ^ '.join(self.variable_strings())})"

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

    def to_string(self) -> str:
        return f"~({' & '.join(self.variable_strings())})"

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

    def to_string(self) -> str:
        return f"~({' | '.join(self.variable_strings())})"

class AllOrNone(Proposition):

    def constraints(self) -> List[GeLineq]:
        return list(
            map(
                lambda vo: GeLineq(
                    valued_variables=list(
                        chain(
                            [
                                ValuedVariable(
                                    id=vo,
                                    value=-len(
                                        set(self.variables) - set([vo]),
                                    ),
                                )
                            ],
                            map(
                                lambda vi: ValuedVariable(
                                    id=vi,
                                    value=1,
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

    def to_string(self) -> str:
        return f"({All(self.variables).to_string()}) | {Nor(self.variables).to_string()}"

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

    def to_string(self) -> str:
        return f"({self.condition.to_string()}) >> ({self.consequence.to_string()})"

    def constraints(self) -> List[GeLineq]:

        if self.condition.is_complex() or self.consequence.is_complex():
            return impl_death_rows(self)

        elif type(self.condition) == And and type(self.consequence) == And:
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                left_union = set(self.condition.variables).intersection(self.consequence.variables).union(self.condition.variables)
                right_union = set(self.consequence.variables).difference(left_union)
                return Impl(
                    And(*left_union),
                    And(*right_union),
                ).constraints()
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
            return Nand(
                *set(self.condition.variables + self.consequence.variables)
            ).constraints()
        elif type(self.condition) == And and type(self.consequence) == Nor:
            # If they share at least one variables, we can reduce directly to a Nand on condition side.
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                return Nand(*self.condition.variables).constraints()
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
            # If they share at least one variables, we can reduce directly to a Nand proposition of the variable's union.
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                return Nand(
                    *set(self.condition.variables).union(self.consequence.variables),
                ).constraints()
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
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                return impl_death_rows(self)
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
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                return impl_death_rows(self)
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
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                return impl_death_rows(self)
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
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                raise Exception("Not implemented")
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
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                return impl_death_rows(self)
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
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                right_union = set(self.condition.variables).intersection(self.consequence.variables).union(self.consequence.variables)
                left_union = set(self.condition.variables).difference(right_union)
                return Impl(
                    And(*right_union),
                    And(*left_union),
                ).constraints()

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
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                return impl_death_rows(self)
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
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                return impl_death_rows(self)
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
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                return impl_death_rows(self)
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
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                return impl_death_rows(self)
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
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                return impl_death_rows(self)
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
            # If they share at least one variables, we can reduce directly to a Nand proposition of the variable's union.
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                return Nand(
                    *set(self.condition.variables).union(self.consequence.variables),
                ).constraints()
            return list(
                set(
                    chain(
                        Impl(
                            And(*self.condition.variables),
                            Or(*self.consequence.variables),
                        ).constraints(),
                        Impl(
                            And(*self.condition.variables),
                            AtMostOne(*self.consequence.variables),
                        ).constraints(),
                    )
                )
            )
        elif type(self.condition) == Or and type(self.consequence) == Xor:
            if len(set(self.condition.variables).union(self.consequence.variables)) < len(self.condition.variables) + len(self.consequence.variables):
                return impl_death_rows(self)
            return list(
                set(
                    chain(
                        Impl(
                            Or(*self.condition.variables),
                            Or(*self.consequence.variables),
                        ).constraints(),
                        Impl(
                            Or(*self.condition.variables),
                            AtMostOne(*self.consequence.variables),
                        ).constraints(),
                    )
                )
            )
        else:
            raise Exception(
                f"Combination of {type(self.condition)} as condition and {type(self.consequence)} as consequence is not yet implemented."
            )

    def atoms(self) -> List[str]:
        return list(
            set(
                chain(
                    self.condition.atoms(),
                    self.consequence.atoms(),
                )
            )
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
