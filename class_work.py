
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

data = {'A': [-3,-1, -1,1,1,3],
        'B': [-4,-2,0,0,2, 4]
        }

df = pd.DataFrame(data,columns=['A','B'])

covMatrix = pd.DataFrame.cov(df)
sn.heatmap(covMatrix, annot=True, fmt='g')
plt.show()
