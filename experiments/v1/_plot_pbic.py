import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
k=np.arange(1,9)
pbic=[18829.6,18649.3,18396.2,18334.1,18400.0,18487.7,18575.1,18662.1]
fig,ax=plt.subplots(figsize=(5,3.2))
ax.plot(k,pbic,'o-',color='#2171b5',markersize=6,linewidth=1.5)
ax.scatter([4],[18334.1],color='#d73027',s=80,zorder=5)
ax.set_xlabel('Number of components ($\\mathcal{P}$)',fontsize=11)
ax.set_ylabel('pBIC',fontsize=11)
ax.set_xticks(k)
ax.grid(True,alpha=0.3,linestyle='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('manuscript/figures/pbic_vs_p.pdf',dpi=200,bbox_inches='tight')
print('Done')
