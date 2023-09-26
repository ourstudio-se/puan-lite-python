import time
import random

from puan_lite import *
from itertools import permutations, product

def generate_random_model(n: int, m: int):
    
    """Generates a random model with n variables"""
    
    # Generate random variables
    perm_gen = permutations(
        list("abcdefghijklmnopqrstuvwxyz"),
        10
    )
    next_variable = lambda _: "".join(next(perm_gen)) 
    variables = [next_variable(_) for _ in range(n)]
    
    # Generate random conjunction
    return Conjunction([
        random.choice([
            And,
            Or,
            Nand,
            Nor,
            Xor,
        ])(
            random.choices(variables, k=random.randint(2, 5))
        ) for _ in range(n)
    ])

import numpy as np
import npycvx
import functools # <- built-in python lib... 
import tqdm

collect = []
for i in tqdm.tqdm(
    range(2,1000)
):
    j = 500
    model = generate_random_model(i,j)

    # Time the conversion
    s1 = time.time()
    ph = model.to_ge_polyhedron()
    e1 = time.time() - s1

    # Some dummy data...
    A = ph.A
    b = ph.b
    objectives = np.random.uniform(size=(1, A.shape[1]))

    # Load solve-function with the now converted numpy
    # matrices/vectors into cvxopt data type...
    solve_part_fn = functools.partial(
        npycvx.solve_lp, 
        *npycvx.convert_numpy(A, b), 
        False,
    )

    # Exectue each objective with solver function
    s2 = time.time()
    solutions = list(
        map(
            solve_part_fn, 
            objectives
        )
    )
    e2 = time.time() - s2
    collect.append({"size": ph.size, "time": e1+e2})

import seaborn as sns
import pandas as pd

df = pd.DataFrame(collect)
plot = sns.lineplot(data=df, x="size", y="time")
fig = plot.get_figure()
fig.savefig("out.png") 