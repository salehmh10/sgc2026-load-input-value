"""Original centered float64 primal Cholesky Ridge solver."""
import gc
import numpy as np
from scipy.linalg import cho_factor, cho_solve

def ridge_statistics(X,y):
    X64=np.asarray(X,dtype=np.float64)
    mean=X64.mean(axis=0); ymean=float(np.mean(y)); X64-=mean
    gram=X64.T@X64
    rhs=X64.T@(y-ymean)
    del X64; gc.collect()
    return mean,ymean,gram,rhs


def ridge_fit(stats,alpha):
    mean,ymean,gram,rhs=stats
    H=gram.copy(order='F'); H.flat[::H.shape[0]+1]+=alpha
    factor=cho_factor(H,lower=True,overwrite_a=True,check_finite=False)
    coef=cho_solve(factor,rhs,check_finite=False)
    residual=gram@coef+alpha*coef-rhs
    relative=float(np.linalg.norm(residual)/max(np.linalg.norm(rhs),1.0))
    assert relative<=1e-7, f'Ridge numerical stationarity failed: {relative}'
    intercept=float(ymean-mean@coef)
    assert np.isfinite(coef).all() and np.isfinite(intercept)
    return (coef,intercept),{'stationarity_relative_residual':relative,'tolerance':1e-7,
                             'solver':'centered float64 Cholesky','coefficient_count':len(coef)}


def predict(family,model,X):
    if family=='Ridge-ARX':
        coef,intercept=model
        return np.asarray(X,dtype=np.float64)@coef+intercept
    return model.predict(X)

