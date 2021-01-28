import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

history = pd.read_csv('history.csv')
molten = pd.melt(history, id_vars=['epoch'])

g = sns.FacetGrid(molten, col='variable', col_wrap=3, sharey=False)
g.map(sns.lineplot, 'epoch', 'value')
g.despine()
g.tight_layout()
plt.savefig("training.png", dpi=200)
plt.close()

if __name__ == '__main__':
    print(molten)