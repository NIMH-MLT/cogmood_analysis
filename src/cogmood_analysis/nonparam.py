import statsmodels.formula.api as smf
import statsmodels.api as sm
import patsy
import numpy as np

# def run_reg_perms(task, tp, ss, dat, perm_indexes):
#     reduced_model = smf.ols(f'{ss} ~ 1 + age + age2 + sex + age:sex + age2:sex', data=dat).fit()
#     residuals_reduced = reduced_model.resid
#     fitted_reduced = reduced_model.fittedvalues
#     full_model = smf.ols(f'{ss} ~ 1 + age + age2 + sex + age:sex + age2:sex + {tp}', data=dat).fit()
#     t0 = full_model.tvalues[tp]
#     perm_dat = dat.copy()
#     t_stars = []
#     t_stars.append(t0)
#     for pid in range(perm_indexes.shape[1]):
#         permuted_residuals = residuals_reduced[perm_indexes[:, pid]].values
#         y_star = fitted_reduced + permuted_residuals
#         perm_dat[ss] = y_star
#         permuted_full_model = smf.ols(f'{ss} ~ 1 + age + age2 + sex + age:sex + age2:sex + {tp}', data=perm_dat).fit()
#         t_stars.append(permuted_full_model.tvalues[tp])
#     t_stars = np.array(t_stars)
#     p_value = np.mean(np.abs(t_stars) >= np.abs(t0))
#     res = dict(
#         task=task,
#         parameter=tp,
#         score=ss,
#         t=t0,
#         perm_p=p_value
#     )
#     for pid in range(perm_indexes.shape[1]):
#         res[f'perm_{pid:04d}']=t_stars[pid]
#     return res

# claude sped up this function for me based on the above input
def run_reg_perms(task, tp, ss, dat, perm_indexes):
    # Fit models once
    reduced_model = smf.ols(f'{ss} ~ 1 + age + age2 + sex + age:sex + age2:sex', data=dat).fit()
    residuals_reduced = reduced_model.resid.values
    fitted_reduced = reduced_model.fittedvalues.values

    formula = f'{ss} ~ 1 + age + age2 + sex + age:sex + age2:sex + {tp}'
    full_model = smf.ols(formula, data=dat).fit()
    t0 = full_model.tvalues[tp]
    partial_r2 = (reduced_model.ssr - full_model.ssr) / reduced_model.ssr
    
    # Pre-build design matrix once
    y, X = patsy.dmatrices(formula, data=dat, return_type='dataframe')
    X_array = X.values
    
    # Find column index for the test parameter
    tp_idx = X.columns.get_loc(tp)
    
    # Vectorized permutation loop
    n_perms = perm_indexes.shape[1]
    t_stars = np.empty(n_perms + 1)
    t_stars[0] = t0
    
    # Pre-compute X'X inverse
    XtX_inv = np.linalg.inv(X_array.T @ X_array)
    
    for pid in range(n_perms):
        # Create permuted y
        y_star = fitted_reduced + residuals_reduced[perm_indexes[:, pid]]
        
        # Direct regression computation: beta = (X'X)^-1 X'y
        beta = XtX_inv @ (X_array.T @ y_star)
        
        # Compute residuals and standard error
        residuals = y_star - X_array @ beta
        mse = np.sum(residuals ** 2) / (len(y_star) - X_array.shape[1])
        se = np.sqrt(mse * XtX_inv[tp_idx, tp_idx])
        
        # t-statistic
        t_stars[pid + 1] = beta[tp_idx] / se
    
    p_value = np.mean(np.abs(t_stars) >= np.abs(t0))
    
    res = dict(
        task=task,
        parameter=tp,
        score=ss,
        t=t0,
        full_r2=full_model.rsquared,
        partial_r2=partial_r2,
        perm_p=p_value
    )
    # for pid in range(n_perms):
    #     res[f'perm_{pid:04d}'] = t_stars[pid]
    
    return res

# def run_reg_boots(task, tp, ss, dat, boot_indexes):
#     full_formula = f'{ss} ~ 1 + age + age2 + sex + age:sex + age2:sex + {tp}'
#     full_model = smf.ols(full_formula, data=dat).fit()

#     reduced_formula = f'{ss} ~ 1 + age + age2 + sex + age:sex + age2:sex'
#     reduced_model = smf.ols(reduced_formula, data=dat).fit()
#     t0 = full_model.tvalues[tp]
#     partial_r2 = (reduced_model.ssr - full_model.ssr) / reduced_model.ssr

#     n_boots = boot_indexes.shape[1]
#     boot_ts = np.empty(n_boots+1)
#     boot_ts[0] = t0
    
#     boot_pr2s = np.empty(n_boots+1)
#     boot_pr2s[0] = partial_r2

#     for bid in range(n_boots):
#         bdat = dat.loc[boot_indexes[:, bid]]
#         boot_reduced_model = smf.ols(reduced_formula, data=bdat).fit()
#         boot_full_model = smf.ols(full_formula, data=bdat).fit()
#         boot_ts[bid + 1] = boot_full_model.tvalues[tp]
#         boot_pr2s[bid + 1] = (boot_reduced_model.ssr - boot_full_model.ssr) / boot_reduced_model.ssr
    
#     boot_t_005, boot_t_025, boot_t_975, boot_t_995 = np.quantile(boot_ts, [0.005, 0.025, 0.975, 0.995])
#     boot_pr2_005, boot_pr2_025, boot_pr2_975, boot_pr2_995 = np.quantile(boot_pr2s, [0.005, 0.025, 0.975, 0.995])
#     res = dict(
#         task=task,
#         parameter=tp,
#         score=ss,
#         t=t0,
#         full_r2=full_model.rsquared,
#         partial_r2=partial_r2,
#         boot_t_mean=boot_ts.mean(),
#         boot_t_std=boot_ts.std(),
#         boot_t_005=boot_t_005,
#         boot_t_025=boot_t_025,
#         boot_t_975=boot_t_975,
#         boot_t_995=boot_t_995,
#         boot_pr2_mean=boot_pr2s.mean(),
#         boot_pr2_std=boot_pr2s.std(),
#         boot_pr2_005=boot_pr2_005,
#         boot_pr2_025=boot_pr2_025,
#         boot_pr2_975=boot_pr2_975,
#         boot_pr2_995=boot_pr2_995
#     )
#     return res

# speed up suggested by cluade based on the above code
def run_reg_boots(task, tp, ss, dat, boot_indexes):
    # Build design matrices once using patsy
    formula_reduced = f'{ss} ~ 1 + age + age2 + sex + age:sex + age2:sex'
    formula_full = f'{ss} ~ 1 + age + age2 + sex + age:sex + age2:sex + {tp}'
    
    y, X_reduced = patsy.dmatrices(formula_reduced, data=dat, return_type='dataframe')
    _, X_full = patsy.dmatrices(formula_full, data=dat, return_type='dataframe')
    
    # Fit original models using matrix API
    reduced_model = sm.OLS(y, X_reduced).fit()
    full_model = sm.OLS(y, X_full).fit()
    
    t0 = full_model.tvalues[tp]
    partial_r2 = (reduced_model.ssr - full_model.ssr) / reduced_model.ssr
    
    n_boots = boot_indexes.shape[1]
    boot_ts = np.empty(n_boots + 1)
    boot_ts[0] = t0
    
    boot_pr2s = np.empty(n_boots + 1)
    boot_pr2s[0] = partial_r2
    
    for bid in range(n_boots):
        # Get the boot indices - these are pandas index labels
        boot_idx = boot_indexes[:, bid]
        
        # Index the design matrices using the pandas index
        y_boot = y.loc[boot_idx]
        X_reduced_boot = X_reduced.loc[boot_idx]
        X_full_boot = X_full.loc[boot_idx]
        
        # Fit models on bootstrapped data
        boot_reduced_model = sm.OLS(y_boot, X_reduced_boot).fit()
        boot_full_model = sm.OLS(y_boot, X_full_boot).fit()
        
        boot_ts[bid + 1] = boot_full_model.tvalues[tp]
        boot_pr2s[bid + 1] = (boot_reduced_model.ssr - boot_full_model.ssr) / boot_reduced_model.ssr
    
    # Compute quantiles once
    boot_t_quantiles = np.quantile(boot_ts, [0.005, 0.025, 0.975, 0.995])
    boot_pr2_quantiles = np.quantile(boot_pr2s, [0.005, 0.025, 0.975, 0.995])
    
    res = dict(
        task=task,
        parameter=tp,
        score=ss,
        t=t0,
        full_r2=full_model.rsquared,
        partial_r2=partial_r2,
        boot_t_mean=boot_ts.mean(),
        boot_t_std=boot_ts.std(),
        boot_t_005=boot_t_quantiles[0],
        boot_t_025=boot_t_quantiles[1],
        boot_t_975=boot_t_quantiles[2],
        boot_t_995=boot_t_quantiles[3],
        boot_pr2_mean=boot_pr2s.mean(),
        boot_pr2_std=boot_pr2s.std(),
        boot_pr2_005=boot_pr2_quantiles[0],
        boot_pr2_025=boot_pr2_quantiles[1],
        boot_pr2_975=boot_pr2_quantiles[2],
        boot_pr2_995=boot_pr2_quantiles[3]
    )
    return res