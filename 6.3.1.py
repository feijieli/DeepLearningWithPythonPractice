# %%
from matplotlib import pyplot as plt
import numpy as np
import os

fname = os.path.join('jena_climate', 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(lines)
# %%

float_data = np.zeros((len(lines), len(header)-1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i] = values
# %%

temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)
# %%
plt.plot(range(1440), temp[:1440])

# %%
