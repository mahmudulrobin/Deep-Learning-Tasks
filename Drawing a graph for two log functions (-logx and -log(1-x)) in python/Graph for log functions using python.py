

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 1.01,0.01)
y=-np.log(x)
z=-np.log(1-x)

plt.plot(x, y, color='b' )
plt.plot(x, z, color='r')


plt.show()