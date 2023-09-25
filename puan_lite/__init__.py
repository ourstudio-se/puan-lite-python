import numpy as np

from dataclasses import dataclass
from typing import List, Union
from itertools import chain
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

@dataclass
class And:

    variables: List[str]

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
class Or:

    variables: List[str]

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
class Xor:

    variables: List[str]

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
class Nand:

    variables: List[str]

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
class Nor:

    variables: List[str]

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

    condition: Union[And, Or, Nand, Nor]
    consequence: Union[And, Or, Nand, Nor]

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
                    bias=(len(self.condition.variables)-1) * len(self.consequence.variables) * -1,
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
                    bias=-1*(len(self.condition.variables)*len(self.consequence.variables)-1),
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
                    bias=-1*(len(self.consequence.variables)*len(self.condition.variables)+len(self.consequence.variables)-1),
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
                    bias=-1*len(self.consequence.variables)*len(self.condition.variables),
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
                                )
                            )
                        ),
                        bias=0,
                    )
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
                                )
                            )
                        ),
                        bias=-1*(len(self.consequence.variables) + len(self.condition.variables) - 1),
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
                                )
                            )
                        ),
                        bias=-1*len(self.consequence.variables),
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
                                )
                            )
                        ),
                        bias=len(self.consequence.variables),
                    ),
                    self.condition.variables,
                )
            )
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
                                )
                            )
                        ),
                        bias=-(len(self.consequence.variables)+1),
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
                    bias=len(self.consequence.variables),
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
                    bias=1,
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
                    bias=-(len(self.consequence.variables)-1),
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
        else:
            raise Exception("Invalid combination of condition and consequence.")

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
        constraints = self.constraints()
        matrix = np.zeros((len(constraints), len(variables)))
        matrix[:, 0] = list(
            map(
                lambda c: c.bias,
                constraints,
            )
        )
        for i, constraint in enumerate(constraints):
            for valued_variable in constraint.valued_variables:
                matrix[i, variables.index(valued_variable.id)] = valued_variable.value

        return ge_polyhedron(
            matrix,
            variables=list(
                map(
                    lambda v: variable(id=v),
                    variables
                )
            ),
        )