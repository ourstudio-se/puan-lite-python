import random

from itertools import product
from puan_lite import *

def test_to_polyhedron_build():
    And(
        Impl(
            And('a', 'b'),
            Or('c', 'd')
        ),
        Impl(
            And('c', 'd'),
            Nand('e', 'f')
        ),
        Xor('e', 'f')
    ).to_ge_polyhedron()

def test_and_proposition():
    variables = ['a', 'b']
    model = And(*variables)
    polyhedron = model.to_ge_polyhedron()
    for interpretation in product([True, False], repeat=len(variables)):
        assert (polyhedron.A.dot(interpretation) >= polyhedron.b).all() == all(interpretation)

    model = And(
        And(*variables),
    )
    polyhedron = model.to_ge_polyhedron()
    for interpretation in product([True, False], repeat=len(variables)):
        assert (polyhedron.A.dot(interpretation) >= polyhedron.b).all() == all(interpretation)

def test_or_proposition():
    variables = ['a', 'b']
    model = And(
        Or(*variables),
    )
    polyhedron = model.to_ge_polyhedron()
    for interpretation in product([True, False], repeat=len(variables)):
        assert (polyhedron.A.dot(interpretation) >= polyhedron.b).all() == any(interpretation)

def test_xor_proposition():
    variables = ['a', 'b']
    model = And(
        Xor(*variables),
    )
    polyhedron = model.to_ge_polyhedron()
    for interpretation in product([True, False], repeat=len(variables)):
        assert (polyhedron.A.dot(interpretation) >= polyhedron.b).all() == (sum(interpretation) % 2 == 1)

def test_nand_proposition():
    variables = ['a', 'b']
    model = And(
        Nand(*variables),
    )
    polyhedron = model.to_ge_polyhedron()
    for interpretation in product([True, False], repeat=len(variables)):
        assert (polyhedron.A.dot(interpretation) >= polyhedron.b).all() == (not all(interpretation))

def test_nor_proposition():
    variables = ['a', 'b']
    model = And(
        Nor(*variables),
    )
    polyhedron = model.to_ge_polyhedron()
    for interpretation in product([True, False], repeat=len(variables)):
        assert (polyhedron.A.dot(interpretation) >= polyhedron.b).all() == (not any(interpretation))

def test_atmostone_proposition():
    variables = ['a', 'b']
    model = And(
        AtMostOne(*variables),
    )
    polyhedron = model.to_ge_polyhedron()
    for interpretation in product([True, False], repeat=len(variables)):
        assert (polyhedron.A.dot(interpretation) >= polyhedron.b).all() == (sum(interpretation) <= 1)

def test_empt_proposition():
    variables = ['a', 'b']
    model = And(
        Empt(*variables),
    )
    polyhedron = model.to_ge_polyhedron()
    for interpretation in product([True, False], repeat=len(variables)):
        assert (polyhedron.A.dot(interpretation) >= polyhedron.b).all() == True

def test_impl_proposition():
    cdvars = ['a', 'b']
    csvars = ['c', 'd']

    prop_code = {
        And: lambda inter: all(inter),
        Or: lambda inter: any(inter),
        Nand: lambda inter: not all(inter),
        Nor: lambda inter: not any(inter),
        Xor: lambda inter: sum(inter) == 1,
        AtMostOne: lambda inter: sum(inter) <= 1,
    }

    prop_combs = [
        (And,   And),
        (And,   Or),
        (And,   Nand),
        (And,   Nor),
        (And,   Xor),
        (And,   AtMostOne),
        (Or,    And),
        (Or,    Or),
        (Or,    Nand),
        (Or,    Nor),
        (Or,    Xor),
        (Or,    AtMostOne),
        (Nand,  And),
        (Nand,  Or),
        (Nand,  Nand),
        (Nand,  Nor),
        (Nor,   And),
        (Nor,   Or),
        (Nor,   Nand),
        (Nor,   Nor)
    ]

    for prop1, prop2 in prop_combs:
        model = And(
            Impl(
                prop1(*cdvars),
                prop2(*csvars),
            )
        )
        polyhedron = model.to_ge_polyhedron()
        for inter in product([True, False], repeat=len(cdvars + csvars)):
            assert (polyhedron.A.dot(inter) >= polyhedron.b).all() == (not prop_code[prop1](inter[:2]) or prop_code[prop2](inter[2:]))

def test_impl_with_same_var_on_both_sides():

    cdvars = ['a', 'b']
    csvars = ['b', 'c']

    prop_code = {
        And: lambda inter: all(inter),
        Or: lambda inter: any(inter),
        Nand: lambda inter: not all(inter),
        Nor: lambda inter: not any(inter),
        Xor: lambda inter: sum(inter) == 1,
        AtMostOne: lambda inter: sum(inter) <= 1,
    }

    prop_combs = [
        (And,   And),
        (And,   Or),
        (And,   Nand),
        (And,   Nor),
        (And,   Xor),
        (And,   AtMostOne),
        (Or,    And),
        (Or,    Or),
        (Or,    Nand),
        (Or,    Nor),
        (Or,    Xor),
        (Nand,  And),
        (Nand,  Or),
        (Nand,  Nand),
        (Nand,  Nor),
        (Nor,   And),
        (Nor,   Or),
        (Nor,   Nand),
        (Nor,   Nor)
    ]

    for prop1, prop2 in prop_combs:
        model = And(
            Impl(
                prop1(*cdvars),
                prop2(*csvars),
            )
        )
        polyhedron = model.to_ge_polyhedron()
        for inter in product([True, False], repeat=len(set(cdvars+csvars))):
            assert (polyhedron.A.dot(inter) >= polyhedron.b).all() == (not prop_code[prop1](inter[:2]) or prop_code[prop2](inter[1:]))

def test_special_char_variables():

    allowed_special_chars = ["!", "@", "#", "$", "%", "&", "*", "(", ")", "-", "+", "=", "[", "]", "{", "}", "|", ":", "'", "\"", ",", "<", ".", ">", "/", "?", "`", "~", "_"]
    random_special_char_variables = list(
        map(
            lambda _: "".join(
                map(
                    lambda _: random.choice(allowed_special_chars),
                    range(10)
                )
            ),
            range(10)
        )
    )
    model = And(*random_special_char_variables)
    assert model.to_ge_polyhedron().shape[1] == len(random_special_char_variables)+1
    assert next(model.solve({random_special_char_variables[0]: 1})).get(random_special_char_variables[0]) == 1

    model = And(
        Impl(
            And(
                Or(
                    *random_special_char_variables[:2],
                ),
                Or(
                    *random_special_char_variables[2:4],
                ),
            ),
            And(
                *random_special_char_variables[4:],
            )
        )
    )
    assert model.to_ge_polyhedron().shape[1] == len(random_special_char_variables)+1
    assert next(model.solve({random_special_char_variables[0]: 1})).get(random_special_char_variables[0]) == 1

def test_special_cases():

    model = And(
        Impl(
            Or(
                And("A","B"),
                And("C","D"),
            ),
            And(
                Xor(*"abc"),
                Xor(*"xyz")
            )
        )
    )

    assert next(model.solve({"A": 1, "B": 1})) == {"A": 1, "B": 1, "C": 0, "D": 0, "a": 1, "b": 0, "c": 0, "x": 1, "y": 0, "z": 0}