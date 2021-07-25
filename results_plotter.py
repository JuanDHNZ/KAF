import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# 1. Nonlinear system 4.2.

path = "results/4.2/"
## a. Testing MSE plot
QKLMS = pd.read_csv(path + "testingMSE_QKLMS_4000-1000_42.csv")
AKB = pd.read_csv(path + "testingMSE_QKLMS_AKB_4000-1000_42.csv")
AMK = pd.read_csv(path + "testingMSE_QKLMS_AMK_4000-1000_42.csv")

plt.figure(figsize=(10,8))
plt.yscale("log")
plt.ylim((1e-30,1e1))
plt.ylabel("TMSE")
plt.xlabel("iterations")
plt.plot(np.linspace(0,4000,800),QKLMS.MSE, color="magenta",label="QKLMS", lw=3, marker="*", ms=10,markevery=20)
plt.plot(np.linspace(0,4000,800),AKB.MSE, color="cyan",label="QKLMS_AKB", lw=3, marker="8", ms=10,markevery=20)
plt.plot(np.linspace(0,4000,800),AMK.MSE, color="lightgreen",label="QKLMS_AMK", lw=3, marker=9, ms=10,markevery=20)
plt.legend()
plt.savefig(path + 'TMSE.png', dpi=300)
plt.show()





