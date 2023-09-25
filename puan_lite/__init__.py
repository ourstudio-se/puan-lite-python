from dataclasses import dataclass
from typing import List, Union
from itertools import chain

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