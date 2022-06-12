import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO 

s = StringIO("""     latency
input     0.0
conv1_1     5.1
relu     2.1
conv1_2    45
relu    2
maxpool    1
conv2_1    20
relu     2
conv2_2     30
relu     2
maxpool     1
conv3_1     20
relu     2
conv3_2    41
relu    2
conv3_3    41
relu     2
conv3_4     43
relu     2
maxpool     1
conv4_1     25
relu     2
conv4_2    48
relu    2
conv4_3    49
relu     2
conv4_4     44
relu     2
maxpool     1
conv5_1     20
relu     2
conv5_2    21
relu    2
conv5_3    22
relu     2
conv5_4     19
relu     2
maxpool     1
linear1     245
relu     2
dropout     1
linear2     18
softmax     1

""")
"""
df = pd.read_csv(s, index_col=0, delimiter=' ', skipinitialspace=True)
ax = df.plot(kind="bar")
ax2 = ax.twinx()
for r in ax.patches[len(df):]:
    r.set_transform(ax2.transData)
ax2.set_ylim(0, 2);
plt.show()
"""
df = pd.read_csv(s, index_col=0, delimiter=' ', skipinitialspace=True)

fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
#ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.8

df.latency.plot(kind='bar', color='green', ax=ax, width=width, position=1)
#df.Size.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

ax.set_ylabel('latency(ms)')
ax.legend(loc='best')
#ax2.set_ylabel('Size(mb)')
#ax2.legend(loc='right')
#plt.legend()
plt.show()

