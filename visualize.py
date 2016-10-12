import seaborn as sns
sns.set_style('white')

import json
import numpy as np

with open("pretrain losses.json") as f:
    data = json.load(f)

print("Data loaded.")

# plot the loss values
sns.plt.plot(data['generator_loss'])
sns.plt.show()

print("Mean loss :", np.mean(data['generator_loss']))
print("Std loss : ", np.std(data['generator_loss']))
print("Min loss : ", np.min(data['generator_loss']))