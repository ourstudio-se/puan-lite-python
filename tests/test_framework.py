from itertools import product
from puan_lite import *

def test_to_polyhedron_build():
    Conjunction([
        Impl(
            And(['a', 'b']),
            Or(['c', 'd'])
        ),
        Impl(
            And(['c', 'd']),
            Nand(['e', 'f'])
        ),
        Xor(['e', 'f'])
    ]).to_ge_polyhedron()

def test_and_proposition():
    variables = ['a', 'b']
    model = Conjunction([
        And(variables),
    ])
    polyhedron = model.to_ge_polyhedron()
    for interpretation in product([True, False], repeat=len(variables)):
        assert (polyhedron.A.dot(interpretation) >= polyhedron.b).all() == all(interpretation)

def test_or_proposition():
    variables = ['a', 'b']
    model = Conjunction([
        Or(variables),
    ])
    polyhedron = model.to_ge_polyhedron()
    for interpretation in product([True, False], repeat=len(variables)):
        assert (polyhedron.A.dot(interpretation) >= polyhedron.b).all() == any(interpretation)

def test_xor_proposition():
    variables = ['a', 'b']
    model = Conjunction([
        Xor(variables),
    ])
    polyhedron = model.to_ge_polyhedron()
    for interpretation in product([True, False], repeat=len(variables)):
        assert (polyhedron.A.dot(interpretation) >= polyhedron.b).all() == (sum(interpretation) % 2 == 1)

def test_nand_proposition():
    variables = ['a', 'b']
    model = Conjunction([
        Nand(variables),
    ])
    polyhedron = model.to_ge_polyhedron()
    for interpretation in product([True, False], repeat=len(variables)):
        assert (polyhedron.A.dot(interpretation) >= polyhedron.b).all() == (not all(interpretation))

def test_nor_proposition():
    variables = ['a', 'b']
    model = Conjunction([
        Nor(variables),
    ])
    polyhedron = model.to_ge_polyhedron()
    for interpretation in product([True, False], repeat=len(variables)):
        assert (polyhedron.A.dot(interpretation) >= polyhedron.b).all() == (not any(interpretation))

def test_atmostone_proposition():
    variables = ['a', 'b']
    model = Conjunction([
        AtMostOne(variables),
    ])
    polyhedron = model.to_ge_polyhedron()
    for interpretation in product([True, False], repeat=len(variables)):
        assert (polyhedron.A.dot(interpretation) >= polyhedron.b).all() == (sum(interpretation) <= 1)

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
        model = Conjunction([
            Impl(
                prop1(cdvars),
                prop2(csvars),
            )
        ])
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
        model = Conjunction([
            Impl(
                prop1(cdvars),
                prop2(csvars),
            )
        ])
        polyhedron = model.to_ge_polyhedron()
        for inter in product([True, False], repeat=len(set(cdvars+csvars))):
            assert (polyhedron.A.dot(inter) >= polyhedron.b).all() == (not prop_code[prop1](inter[:2]) or prop_code[prop2](inter[1:]))