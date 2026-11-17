"""STA9890 residential assessment valuation (2025), rebuilt 2026.

Run locally: python train_models.py
Reads the seven competition CSVs from ./data/ by default.
Competition data only; no external targets/labels.
The deterministic row-order split recreates the 2026 EDA holdout on the original CSVs.
"""
import os, json, gc, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv('STA9890_DATA_DIR', str(REPO_ROOT / 'data')))
OUT = Path(os.getenv('STA9890_STEP3_OUT', str(REPO_ROOT / 'outputs')))
OUT.mkdir(parents=True,exist_ok=True)
SEED = 42

def say(*a): print(pd.Timestamp.now().strftime('%H:%M:%S'), *a,flush=True)
def rmse(y, pred): return float(np.sqrt(np.mean(np.square(np.asarray(y,dtype=np.float64)-np.asarray(pred,dtype=np.float64)))))
def coerce_float(s): return pd.to_numeric(s,errors='coerce').astype('float32')
def div(a,b): return a/b.replace(0,np.nan)
def downcast(df):
    for c in df.select_dtypes(include=['float64','int64']).columns:
        df[c] = pd.to_numeric(df[c], downcast='float') if df[c].dtype=='float64' else pd.to_numeric(df[c],downcast='integer')
    return df

def prepare():
    say('Read assessment train/test')
    tr=pd.read_csv(DATA/'assessment_history_train.csv',low_memory=False)
    te=pd.read_csv(DATA/'assessment_history_test.csv',low_memory=False)
    assert tr.acct.is_unique and te.acct.is_unique
    assert not tr.acct.isin(te.acct).any()
    assert tr.TARGET.notna().all()
    assert 'TARGET' not in te.columns
    tr['is_test']=np.uint8(0); te['is_test']=np.uint8(1)
    y=tr.TARGET.to_numpy(dtype='float32')
    ids_tr=tr.acct.copy(); ids_te=te.acct.copy()
    # Never include train-only columns in predictors.
    excluded={'assessed_2019','building_value_2019','land_value_2019','TARGET'}
    feature_cols=[x for x in tr.columns if x not in excluded]
    assert set(feature_cols) == set(te.columns)
    x=pd.concat([tr[feature_cols],te[feature_cols]],ignore_index=True)
    del tr,te;gc.collect()
    say('Assessment combined',x.shape)
    # For groups, keep raw year features and create temporal dynamics.
    for prefix in ['assessed','building_value','land_value','building_area','land_area']:
        years=[2015,2016,2017,2018] + ([2019] if prefix.endswith('area') else [])
        cols=[f'{prefix}_{yr}' for yr in years]
        allvals=x[cols].apply(pd.to_numeric,errors='coerce').astype('float32')
        x[f'{prefix}_last']=allvals.iloc[:,::-1].bfill(axis=1).iloc[:,0].astype('float32')
        x[f'{prefix}_n_observed']=allvals.notna().sum(axis=1).astype('int8')
        x[f'{prefix}_mean']=allvals.mean(axis=1).astype('float32')
        x[f'{prefix}_std']=allvals.std(axis=1).astype('float32')
        x[f'{prefix}_min']=allvals.min(axis=1).astype('float32')
        x[f'{prefix}_max']=allvals.max(axis=1).astype('float32')
        x[f'{prefix}_last_minus_mean']=(allvals.iloc[:,-1]-allvals.mean(axis=1)).astype('float32')
        x[f'{prefix}_delta_17_18']=(x[f'{prefix}_2018']-x[f'{prefix}_2017']).astype('float32')
        x[f'{prefix}_delta_16_17']=(x[f'{prefix}_2017']-x[f'{prefix}_2016']).astype('float32')
        x[f'{prefix}_delta_15_16']=(x[f'{prefix}_2016']-x[f'{prefix}_2015']).astype('float32')
        x[f'{prefix}_growth_17_18']=div(x[f'{prefix}_delta_17_18'],coerce_float(x[f'{prefix}_2017'])).clip(-2,5).astype('float32')
        x[f'{prefix}_growth_16_17']=div(x[f'{prefix}_delta_16_17'],coerce_float(x[f'{prefix}_2016'])).clip(-2,5).astype('float32')
        # The interval endpoint is fixed, treating absent historical information explicitly.
        x[f'{prefix}_delta_15_18']=(x[f'{prefix}_2018']-x[f'{prefix}_2015']).astype('float32')
        del allvals
    x['base_last_assessed']=x[[f'assessed_{z}' for z in [2018,2017,2016,2015]]].bfill(axis=1).iloc[:,0].astype('float32')
    x['base_missing']=x.base_last_assessed.isna().astype('int8')
    x['last_assessed_year'] = 2018.0
    x.loc[x.assessed_2018.isna(),'last_assessed_year'] = np.nan
    for z in [2017,2016,2015]:
        x['last_assessed_year'] = x.last_assessed_year.fillna(pd.Series(np.where(x[f'assessed_{z}'].notna(),z,np.nan), index=x.index))
    x['last_assessed_year']=x.last_assessed_year.astype('float32')
    x['assessed_minus_components_2018']=(x.assessed_2018-x.building_value_2018-x.land_value_2018).astype('float32')
    x['assessed_over_components_2018']=div(x.assessed_2018,coerce_float(x.building_value_2018+x.land_value_2018)).clip(0,10).astype('float32')
    x['building_fraction_2018']=div(x.building_value_2018,coerce_float(x.building_value_2018+x.land_value_2018)).clip(0,5).astype('float32')
    x['value_per_land_area_2018']=div(x.land_value_2018,coerce_float(x.land_area_2018)).clip(0,1e6).astype('float32')
    x['value_per_building_area_2018']=div(x.building_value_2018,coerce_float(x.building_area_2018)).clip(0,1e6).astype('float32')
    x['protest_history_count']=x[[f'protested_{z}' for z in range(2015,2019)]].sum(axis=1).astype('int8')
    x['building_area_change_18_19']=(x.building_area_2019-x.building_area_2018).astype('float32')
    x['land_area_change_18_19']=(x.land_area_2019-x.land_area_2018).astype('float32')
    x['building_area_change_rate_18_19']=div(x.building_area_change_18_19,coerce_float(x.building_area_2018)).clip(-2,5).astype('float32')
    x['land_area_change_rate_18_19']=div(x.land_area_change_18_19,coerce_float(x.land_area_2018)).clip(-2,5).astype('float32')
    # 2019 building details are available for BOTH training and test in this particular competition.
    usecols=['acct','floor_area_primary','floor_area_upper','floor_area_lower','garage_area','porch_area','deck_area',
             'mobile_home_area','floors','half_bath','full_bath','total_rooms','bedrooms','fireplaces','elevator','quality',
             'quality_description','year_built','year_remodeled','building_condition','foundation_type','grade',
             'has_cooling','has_heat','physical_condition','exterior_walls']
    say('Read 2019 building details')
    b19=pd.read_csv(DATA/'building_details_2019.csv', usecols=usecols,low_memory=False)
    assert b19.acct.is_unique
    for col in b19.columns:
        if b19[col].dtype=='int64': b19[col]=pd.to_numeric(b19[col],downcast='integer')
    # rename all attributes, make feature origin explicit.
    b19=b19.rename(columns={z:'b19_'+z for z in b19.columns if z!='acct'})
    x=x.merge(b19,on='acct',how='left',validate='1:1',sort=False,copy=False)
    del b19;gc.collect()
    say('Read 2018 selected building details')
    b18=pd.read_csv(DATA/'building_details_2018.csv',usecols=['acct','floor_area_primary','floor_area_upper','floor_area_lower',
                                 'garage_area','porch_area','total_rooms','bedrooms','quality','year_built','year_remodeled',
                                 'building_condition','grade','physical_condition'],low_memory=False)
    assert b18.acct.is_unique
    for col in b18.columns:
        if b18[col].dtype=='int64': b18[col]=pd.to_numeric(b18[col],downcast='integer')
    b18=b18.rename(columns={z:'b18_'+z for z in b18.columns if z!='acct'})
    x=x.merge(b18,on='acct',how='left',validate='1:1',sort=False,copy=False)
    del b18;gc.collect()
    for col in ['floor_area_primary','floor_area_upper','floor_area_lower','garage_area','porch_area','bedrooms','total_rooms']:
        a=coerce_float(x[f'b19_{col}']);b=coerce_float(x[f'b18_{col}']);
        x[f'b_delta_{col}']=(a-b).astype('float32')
        if col in ['floor_area_primary','floor_area_upper','floor_area_lower']:
            x[f'b_rate_{col}']=div(a-b,b).clip(-2,5).astype('float32')
    for col in ['quality','grade','building_condition','physical_condition']:
        x[f'b_changed_{col}']= (x[f'b19_{col}'] != x[f'b18_{col}']).astype('int8')
    for z in [18,19]:
        q=f'b{z}_year_remodeled';r=pd.to_numeric(x[q], errors='coerce')
        x[q+'_valid']=r.where(r.between(1800,2019),np.nan).astype('float32')
        x[q+'_recent']=r.between(2015,2019).astype('int8')
        by=f'b{z}_year_built';p=pd.to_numeric(x[by], errors='coerce')
        x[by+'_age']=(2019-p.where(p.between(1700,2019),np.nan)).astype('float32')
    x['b_recent_remodel_change']=(x.b19_year_remodeled_valid-x.b18_year_remodeled_valid).astype('float32')
    x['b_floor_area_total_19']=x[['b19_floor_area_primary','b19_floor_area_upper','b19_floor_area_lower']].sum(axis=1,min_count=1).astype('float32')
    x['b_floor_area_to_land_19']=div(x.b_floor_area_total_19,coerce_float(x.land_area_2019)).clip(0,100).astype('float32')
    x['b_total_bathrooms_19']=(x.b19_full_bath+0.5*x.b19_half_bath).astype('float32')
    # Strip identifiers other than geographical labels and explicit categorical features.
    x=x.drop(columns=['acct','is_test'])
    cat=['region','school_dist','zone','subneighborhood','neighborhood']
    for z in ['19','18']:
        cat += [f'b{z}_{v}' for v in ['quality','grade','building_condition','physical_condition']]
    cat += ['b19_quality_description','b19_foundation_type','b19_exterior_walls']
    for c in cat:
        x[c]=x[c].astype('category')
    for col in x.columns:
        if col not in cat:
            if x[col].dtype == 'object':
                x[col]=x[col].astype('category');cat.append(col)
            elif pd.api.types.is_bool_dtype(x[col]): x[col]=x[col].astype('int8')
            elif str(x[col].dtype)=='float64': x[col]=x[col].astype('float32')
            elif str(x[col].dtype)=='int64': x[col]=pd.to_numeric(x[col],downcast='integer')
    # Hard fail for leaking any 2019 target/component columns.
    assert not {'TARGET','assessed_2019','building_value_2019','land_value_2019'}.intersection(x.columns)
    assert not any(c.startswith('b19_building_value') or c.startswith('b19_land_value') for c in x.columns)
    assert x.shape[0] == len(ids_tr)+len(ids_te)
    # 'training' and 'competition test' share exact feature generation, with same categorical codes.
    n=len(ids_tr);xtr=x.iloc[:n].copy();xte=x.iloc[n:].copy()
    del x;gc.collect()
    say('Prepared data',xtr.shape, xte.shape, 'feature categories:',len(cat), 'size MB:',round(xtr.memory_usage(deep=True).sum()/1e6))
    return xtr,xte,y,ids_tr,ids_te,cat

def cohort_summary(y, p, missing):
    e=(np.asarray(p,dtype=np.float64)-np.asarray(y,dtype=np.float64))**2
    return {'rmse_all':float(np.sqrt(e.mean())),
            'rmse_with_history':float(np.sqrt(e[~missing].mean())),
            'rmse_no_history':float(np.sqrt(e[missing].mean())),
            'n_with_history':int((~missing).sum()),'n_no_history':int(missing.sum()),
            'no_history_share_squared_error_pct':float(100*e[missing].sum()/e.sum()),
            'high_target_gt_1m_rmse':float(np.sqrt(e[np.asarray(y)>1e6].mean())),
            'high_target_gt_1m_n':int((np.asarray(y)>1e6).sum())}

def main():
    t0=time.time()
    xtr,xte,y,ids_tr,ids_te,cat=prepare()
    # Recreate the exact 2026 validation split from original file order.
    # Do not commit the ID-level split file: account IDs are competition data.
    # If the input CSV order changes, use the original order or supply an approved split mapping.
    _, idx_val = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED)
    is_val=np.zeros(len(y),dtype=bool)
    is_val[idx_val]=True
    assert is_val.sum()==125658 and (~is_val).sum()==502629
    assert ids_tr.is_unique
    # Use the holdout for major model comparisons, tune tree iterations INSIDE development.
    xdev=xtr.loc[~is_val].reset_index(drop=True)
    xval=xtr.loc[is_val].reset_index(drop=True)
    ydev=y[~is_val];yval=y[is_val]
    missing=xval.base_missing.to_numpy().astype(bool)
    baseline_median=np.median(ydev)
    base=xval.base_last_assessed.to_numpy().copy()
    base[np.isnan(base)]=baseline_median
    baseline=cohort_summary(yval,base,missing)
    say('Baseline',baseline)
    idx_a,idx_b=train_test_split(np.arange(len(ydev)),test_size=0.12,random_state=SEED)
    inner_data=(xdev.iloc[idx_a],ydev[idx_a],xdev.iloc[idx_b],ydev[idx_b])
    # no fillna categorical; LightGBM natively handles missing and category IDs
    params={'objective':'regression','metric':'rmse','learning_rate':0.035,'num_leaves':47,
            'max_depth':-1,'min_child_samples':60,'colsample_bytree':0.85,'subsample':0.85,
            'subsample_freq':1,'reg_lambda':7.0,'n_estimators':1800,'verbosity':-1,
            'random_state':SEED,'n_jobs':4,'max_bin':255}
    result={'baseline':baseline,'settings':{'outer_validation_n':len(yval),'development_n':len(ydev),
            'seed':SEED,'use_2019_protest':True,'uses_2019_building_details':True},'models':{}}
    preds={}
    models={}
    for targetkind in ['direct','residual']:
        say('Train LightGBM',targetkind,'inner CV tuning')
        ya, yb=inner_data[1],inner_data[3]
        a,b=inner_data[0],inner_data[2]
        if targetkind=='residual':
            # Training-only median is stable for imputing absence of historical value.
            ba=a.base_last_assessed.fillna(float(np.median(ydev))).to_numpy(dtype='float32')
            bb=b.base_last_assessed.fillna(float(np.median(ydev))).to_numpy(dtype='float32')
            ya=ya-ba;yb=yb-bb
        m=lgb.LGBMRegressor(**params)
        m.fit(a,ya,eval_set=[(b,yb)],eval_metric='rmse',callbacks=[lgb.early_stopping(95,verbose=False)])
        best_iter=int(m.best_iteration_)
        say('Best iter', targetkind,best_iter,'inner best score',round(float(m.best_score_['valid_0']['rmse']),2))
        del m
        p1=params.copy(); p1['n_estimators']=best_iter
        model=lgb.LGBMRegressor(**p1)
        if targetkind=='residual':
            dd=xdev.base_last_assessed.fillna(float(np.median(ydev))).to_numpy(dtype='float32')
            yy=ydev-dd
        else: yy=ydev
        model.fit(xdev,yy)
        pv=model.predict(xval)
        if targetkind=='residual':
            pv=pv+base
        pv=np.maximum(pv,0)
        preds[targetkind]=pv
        models[targetkind]=model
        score=cohort_summary(yval,pv,missing)
        result['models'][targetkind]={'best_iteration':best_iter,'inner_score':float(model.best_score_['training']['rmse']) if hasattr(model,'best_score_') and 'training' in model.best_score_ else None, 'score':score}
        pd.DataFrame({'feature':model.feature_name_,'gain':model.booster_.feature_importance(importance_type='gain'), 'split':model.booster_.feature_importance(importance_type='split')}).sort_values('gain',ascending=False).to_csv(OUT/f'importance_{targetkind}.csv',index=False)
        say('HOLDOUT',targetkind,score)
        gc.collect()
    # Compare blending after models trained; no extra fit.
    for w in [0.25,0.5,0.75]:
        pred=(1-w)*preds['direct']+w*preds['residual']
        score=cohort_summary(yval,pred,missing)
        result['models'][f'blend_residual_weight_{w}']={'score':score}
        say('HOLDOUT blend',w,round(score['rmse_all'],2))
    # Additional one-parameter segment-specific combinations for diagnostics only.
    best=min(result['models'],key=lambda z: result['models'][z]['score']['rmse_all'])
    say('BEST overall',best,'RMSE',result['models'][best]['score']['rmse_all'])
    if best == 'direct': pval=preds['direct']
    elif best == 'residual':pval=preds['residual']
    else:
        w=float(best.split('_')[-1]);pval=(1-w)*preds['direct']+w*preds['residual']
    pd.DataFrame({'acct':ids_tr.loc[is_val].to_numpy(),'truth':yval,'prediction':pval,
      'historical_assessed_2018':xval.assessed_2018.to_numpy(),'has_history':~missing}).to_csv(OUT/'holdout_predictions.csv.gz',index=False,compression='gzip')
    del xval,xdev,inner_data;gc.collect()
    # Final competition output: all labels, no use of test labels; same picked recipe and number of boosting rounds.
    say('Refit chosen recipe on all labelled 2019 properties for test prediction')
    selected=[best] if best in ['direct','residual'] else ['direct','residual']
    final_test_preds={}
    for k in selected:
        p=params.copy();p['n_estimators']=result['models'][k]['best_iteration']
        model=lgb.LGBMRegressor(**p)
        if k=='residual':
            final_base=xtr.base_last_assessed.fillna(float(np.median(y))).to_numpy(dtype='float32')
            yy=y-final_base
        else:yy=y
        model.fit(xtr,yy)
        pt=model.predict(xte)
        if k=='residual':pt=pt+xte.base_last_assessed.fillna(float(np.median(y))).to_numpy(dtype='float32')
        final_test_preds[k]=np.maximum(pt,0)
        model.booster_.save_model(str(OUT/f'final_{k}.txt'))
        say('Saved final',k)
        del model;gc.collect()
    if best in ['direct','residual']:ptest=final_test_preds[best]
    else:
        w=float(best.split('_')[-1]); ptest=(1-w)*final_test_preds['direct']+w*final_test_preds['residual']
    assert len(ptest)==len(ids_te) and np.isfinite(ptest).all() and (ptest>=0).all()
    pd.DataFrame({'ACCOUNT':ids_te.to_numpy(),'TARGET':ptest}).to_csv(OUT/'submission.csv',index=False)
    # Record methodology and score, no private test labels available; this is validation score, not Kaggle result.
    result['selected']=best;result['validation_rmse']=result['models'][best]['score']['rmse_all']
    result['elapsed_seconds']=time.time()-t0
    result['n_features']=len(xtr.columns)
    result['categoricals']=cat
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    say('DONE',json.dumps({'selected':best,'rmse':result['validation_rmse'],'seconds':round(result['elapsed_seconds'],1),'features':result['n_features']}))

if __name__=='__main__':main()
