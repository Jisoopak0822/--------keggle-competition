from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
REPO_ROOT=Path(__file__).resolve().parents[1]
P=Path(os.getenv('STA9890_STEP3_OUT', str(REPO_ROOT/'outputs')))
P.mkdir(parents=True, exist_ok=True)
FIGS=REPO_ROOT/'figures'
FIGS.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.dpi':160,'font.size':11})

models={'Carry-forward baseline':56756.82,'LightGBM direct target':45404.28,'Direct + residual 50:50':33420.65,'LightGBM residual (best)':29775.36}
fig,ax=plt.subplots(figsize=(9,4.7));names=list(models);v=list(models.values());bars=ax.barh(names, v)
ax.invert_yaxis();ax.set_xlim(0,65000);ax.set_xlabel('RMSE (USD; lower is better)')
ax.set_title('Model validation on 125,658 held-out properties')
for bar,vv in zip(bars,v):ax.text(vv+400,bar.get_y()+bar.get_height()/2,f'{vv:,.0f}',va='center',fontsize=9)
fig.tight_layout();fig.savefig(FIGS/'01_model_comparison.png',bbox_inches='tight');plt.close(fig)

imp=pd.read_csv(P/'importance_residual.csv').head(14).sort_values('gain')
imp['relative']=100*imp['gain']/pd.read_csv(P/'importance_residual.csv')['gain'].sum()
fig,ax=plt.subplots(figsize=(9,6.8));ax.barh(imp.feature,imp.relative)
ax.set_xlabel('Share of total LightGBM gain (%)')
ax.set_title('Top features — residual prediction model')
fig.tight_layout();fig.savefig(FIGS/'02_feature_importance.png',bbox_inches='tight');plt.close(fig)

pred=pd.read_csv(P/'holdout_predictions.csv.gz')
# Display an interpretable range; extreme outliers are reviewed in metrics instead.
s=pred[(pred.truth>0)&(pred.truth<1_000_000)&(pred.prediction>=0)&(pred.prediction<1_000_000)].sample(n=8000,random_state=42)
fig,ax=plt.subplots(figsize=(7,6.5));ax.scatter(s.truth/1000,s.prediction/1000,s=6,alpha=.12)
ax.plot([0,1000],[0,1000],ls='--',lw=1);ax.set_xlim(0,1000);ax.set_ylim(0,1000)
ax.set(xlabel='Actual 2019 assessment (thousands USD)',ylabel='Predicted 2019 assessment (thousands USD)',title='Held-out predictions (sample, under $1m)')
fig.tight_layout();fig.savefig(FIGS/'03_predicted_vs_actual.png',bbox_inches='tight');plt.close(fig)

seg=[]
for label,mask in [('History available',pred.has_history),('No history',~pred.has_history),('High value >$1m',pred.truth>1e6)]:
  temp=pred[mask]
  score=np.sqrt(np.square(temp.truth-temp.prediction).mean())
  seg.append((label,len(temp),score))
fig,ax=plt.subplots(figsize=(8,4.5));labs=[z[0] for z in seg];values=[z[2] for z in seg]; bars=ax.bar(labs, values)
for bar,(_,n,v) in zip(bars,seg): ax.text(bar.get_x()+bar.get_width()/2,v+2500,f'{v:,.0f}\n(n={n:,})',ha='center',fontsize=9)
ax.set_ylim(0,max(values)*1.25);ax.set_ylabel('RMSE (USD)');ax.set_title('Where prediction errors remain high')
fig.tight_layout();fig.savefig(FIGS/'04_segment_errors.png',bbox_inches='tight');plt.close(fig)
print('done figures')
