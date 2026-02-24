from grid import make_default_grid

g = make_default_grid()
print(len(g.maturities), "maturities")
print(len(g.k_grid), "k points")
print("k range:", g.k_grid[0], g.k_grid[-1])

Ks = g.strikes_from_forward(100.0)
print("K range:", Ks[0], Ks[-1])