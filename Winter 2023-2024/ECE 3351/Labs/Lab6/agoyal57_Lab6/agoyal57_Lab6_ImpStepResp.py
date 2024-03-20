import numpy as np

# define the analytic dt system functions
def sys1(n):
    res = 0.5 * (0.8) ** n + (0.7) ** n + 2 * (0.4) ** n
    return res
def sys2(n):
    res = 2.236 * (np.sqrt(2)/2) ** n * np.cos(0.463 + 0.785 * n) + 0.5 * (-0.7) ** n
    return res

# func to calculate impulse response
def impulse(system, iterations):
    resp = []
    for n in range(iterations):
        val = system(n)
        val_round = float(f'{val:.3f}')
        resp.append(val_round)
    return resp

# func to calculate step response
def step(system, iterations):
    resp = []
    imp = impulse(system=system, iterations=iterations)
    
    # the first element in step response is same as impulse resp
    resp.append(imp[0])
    for n in range(1, iterations):
        val = resp[n-1] + imp[n]
        val_round = float(f'{val:.2f}')
        resp.append(val_round)
    return resp

# func to get impulse and step response written in a file
def fileWrite(file, system, label):
    file.write(f'{label.upper()} - IMPULSE RESPONSE\n')
    file.write(str(impulse(system, 10)))
    file.write('\n')
    file.write(f'{label.upper()} - STEP RESPONSE\n')
    file.write(str(step(system, 10)))
    file.write('\n')

## BEGIN MAIN
file = open('responses.txt', 'w')
fileWrite(file, sys1, 'system 1')
fileWrite(file, sys2, 'system 2')
file.close()
## END MAIN