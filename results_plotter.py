import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib
sns.set()



path = "results/4.2/New/learning_curves/"

filts = ["cb_{}_4.2_AKB_5003.csv".format(fil) for fil in ["QKLMS","QKLMS_AKB","QKLMS_AMK"]]

QKLMS = pd.read_csv(path + filts[0])
AKB = pd.read_csv(path + filts[1])
AMK = pd.read_csv(path + filts[2])


#%% Nonlinear system 4.2. New TMSE results for best CB criteria

plt.figure(figsize=(10,8))
plt.yscale("log")
plt.ylim((1e-1,0.2e1))
plt.ylabel("TMSE")
plt.xlabel("iterations")
plt.plot(np.linspace(0,4000,len(QKLMS.mean_TMSE)),QKLMS.mean_TMSE, color="magenta",label="QKLMS", lw=3, marker="*", ms=10,markevery=20)
plt.plot(np.linspace(0,4000,len(AKB.mean_TMSE)),AKB.mean_TMSE, color="cyan",label="QKLMS_AKB", lw=3, marker="8", ms=10,markevery=20)
plt.plot(np.linspace(0,4000,len(AMK.mean_TMSE)),AMK.mean_TMSE, color="lightgreen",label="QKLMS_AMK", lw=3, marker=9, ms=10,markevery=20)
plt.legend()
plt.savefig(path + 'TMSE_minCB.png', dpi=300)
tikzplotlib.save(path + 'TMSE_minCB.tex')
plt.show()


#%% Nonlinear system 4.2. New CB growth for best CB criteria

plt.figure(figsize=(10,8))
plt.ylabel("TMSE")
plt.xlabel("iterations")
plt.plot(np.linspace(0,4000,len(QKLMS.mean_TMSE)),QKLMS.mean_CB, color="magenta",label="QKLMS", lw=3, marker="*", ms=10,markevery=20)
plt.plot(np.linspace(0,4000,len(AKB.mean_TMSE)),AKB.mean_CB, color="cyan",label="QKLMS_AKB", lw=3, marker="8", ms=10,markevery=20)
plt.plot(np.linspace(0,4000,len(AMK.mean_TMSE)),AMK.mean_CB, color="lightgreen",label="QKLMS_AMK", lw=3, marker=9, ms=10,markevery=20)
plt.legend()
plt.savefig(path + 'CB_minCB.png', dpi=300)
tikzplotlib.save(path + 'CB_minCB.tex')
plt.show()
