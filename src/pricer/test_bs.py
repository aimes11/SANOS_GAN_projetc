from bs import BSInputs, price, vega, forward, discount


inp = BSInputs(spot = 100, strike = 100, ttm = 1.0, rate = 0.05, div = 0.0, vol = 0.05)

print('Call price', price(inp, 'call'))
print('Put price', price(inp, 'put'))
print("Vega", vega(inp))
print('Inputs', inp)

F = forward(inp.spot, inp.rate, inp.div, inp.ttm)
D = discount(inp.rate, inp.div, inp.ttm)
lhs = price(inp, 'call') - price(inp, 'put')
rhs = D * (F - inp.strike)
print("Parity error", lhs - rhs)
