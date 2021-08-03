import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib
sns.set()



path = "../results/4.2/New/learning_curves/"

#State of the art
filts = ["cb_{}_4.2_AKB_5003.csv".format(fil) for fil in ["QKLMS","QKLMS_AKB"]]
QKLMS = pd.read_csv(path + filts[0])
AKB = pd.read_csv(path + filts[1])

#Proposed
amks = ["cb2_QKLMS_AMK_4.2_AKB_5003_K_{}.csv".format(k) for k in [1,2,4,6,8]]
amk_k1 = pd.read_csv(path + amks[0])
amk_k2 = pd.read_csv(path + amks[1])
amk_k4 = pd.read_csv(path + amks[2])
amk_k6 = pd.read_csv(path + amks[3])
amk_k8 = pd.read_csv(path + amks[4])





#%% Nonlinear system 4.2. New TMSE results for best CB criteria

plt.figure(figsize=(16,9))
plt.yscale("log")
plt.ylim((0.5,0.1e1))
plt.ylabel("TMSE")
plt.xlabel("iterations")
plt.plot(np.linspace(0,4000,len(QKLMS.mean_TMSE)),QKLMS.mean_TMSE, color="m",label="QKLMS", lw=2, marker="*", ms=10,markevery=10)
plt.plot(np.linspace(0,4000,len(AKB.mean_TMSE)),AKB.mean_TMSE, color="c",label="QKLMS_AKB", lw=2, marker="8", ms=10,markevery=10)
plt.plot(np.linspace(0,4000,len(amk_k1.mean_TMSE)),amk_k1.mean_TMSE, color="tab:blue",label="Proposed K=1", lw=2, marker="o", ms=10,markevery=10)
plt.plot(np.linspace(0,4000,len(amk_k2.mean_TMSE)),amk_k2.mean_TMSE, color="tab:red",label="Proposed K=2", lw=2, marker="+", ms=10,markevery=10)
plt.plot(np.linspace(0,4000,len(amk_k4.mean_TMSE)),amk_k4.mean_TMSE, color="tab:pink",label="Proposed K=4", lw=2, marker="x", ms=10,markevery=10)
plt.plot(np.linspace(0,4000,len(amk_k6.mean_TMSE)),amk_k6.mean_TMSE, color="tab:green",label="Proposed K=6", lw=2, marker="X", ms=10,markevery=10)
plt.plot(np.linspace(0,4000,len(amk_k8.mean_TMSE)),amk_k8.mean_TMSE, color="tab:orange",label="Proposed K=8", lw=2, marker=9, ms=10,markevery=10)
plt.legend()
plt.savefig('TMSE_minCB.png', dpi=300)
tikzplotlib.save('TMSE_minCB.tex')
plt.show()


#%% Nonlinear system 4.2. New CB growth for best CB criteria

# plt.figure(figsize=(10,8))
# plt.ylabel("TMSE")
# plt.xlabel("iterations")
# plt.plot(np.linspace(0,4000,len(QKLMS.mean_TMSE)),QKLMS.mean_CB, color="magenta",label="QKLMS", lw=3, marker="*", ms=10,markevery=20)
# plt.plot(np.linspace(0,4000,len(AKB.mean_TMSE)),AKB.mean_CB, color="cyan",label="QKLMS_AKB", lw=3, marker="8", ms=10,markevery=20)
# plt.plot(np.linspace(0,4000,len(AMK.mean_TMSE)),AMK.mean_CB, color="lightgreen",label="QKLMS_AMK", lw=3, marker=9, ms=10,markevery=20)
# plt.legend()
# plt.savefig(path + 'CB_minCB.png', dpi=300)
# tikzplotlib.save(path + 'CB_minCB.tex')
# plt.show()
