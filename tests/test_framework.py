from puan_lite import *

def test_to_polyhedron():
    
    model = Conjunction([
        Impl(
            And(['a', 'b']),
            Or(['c', 'd'])
        ),
        Impl(
            And(['c', 'd']),
            Nand(['e', 'f'])
        ),
        Xor(['e', 'f'])
    ])
    ph = model.to_ge_polyhedron()
    x=1