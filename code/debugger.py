
def pv(variables):
  print([var.__name__ + ' = ' + str(var) for var in variables].join(', '))
