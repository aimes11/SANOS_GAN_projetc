from bs import BSInputs, price
from implied_vol import implied_vol

inp =BSInputs(spot = 100, strike = 100, ttm = 1.0, rate = 0.05, div = 0.0, vol = 0.2)
mkt_price = price(inp, 'call')

inp_guess = BSInputs(spot = 100, strike = 100, ttm = 1.0, rate = 0.05, div = 0.0, vol = 0.3)
iv = implied_vol(mkt_price, inp_guess, 'call')
print("Market price", mkt_price)
print("Implied vol", iv)