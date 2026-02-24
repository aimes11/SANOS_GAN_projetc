from src.market.grid import make_default_grid
from src.market.synthetic_surface import build_price_surface
from src.pricer.bs import forward
from src.validation.butterfly import check_butterfly

grid = make_default_grid()

spot, r, q = 100.0, 0.05, 0.02
price_surf = build_price_surface(grid, spot=spot, rate=r, div=q)

# build strikes surface aligned with the grid (for each maturity, use its forward)
strikes_surf = []
for T in grid.maturities:
    F = forward(spot, T, r, q)
    strikes_surf.append(grid.strikes_from_forward(F))

rep = check_butterfly(price_surf, strikes_surf, eps=1e-12)
print(rep)