from base_utils.dir_util import mkdirs
import seaborn

color_keys = seaborn.colors.xkcd_rgb.keys()
print(type(color_keys))

import matplotlib.pyplot as plt
import seaborn
# import matplotlib
# matplotlib.use('TkAgg')
import math
math.ceil()
color_key = [color for color in seaborn.colors.xkcd_rgb.keys()]
figure1 = plt.figure()

for i in range(30):
    ax = figure1.add_subplot(8, 8, i+1, aspect='equal')
    ax.add_patch(plt.Rectangle((0, 0), 1, 2, color=seaborn.xkcd_rgb[color_key[i]], alpha=1))
    # ax1.text(0, 0, '12312313', fontsize=30)
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()