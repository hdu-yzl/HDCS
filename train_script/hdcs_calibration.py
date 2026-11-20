"""Positive-slope Platt calibration fitted exclusively to validation logits."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_affine(logits, labels):
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    def objective(parameters):
        transformed = parameters[0]*logits + parameters[1]
        residual = expit(transformed)-labels
        return (float(np.mean(np.logaddexp(0., transformed)-labels*transformed)),
                np.array([np.mean(residual*logits), np.mean(residual)]))
    fitted = minimize(objective, [1., 0.], jac=True, method='L-BFGS-B',
                      bounds=[(.05, 4.), (-5., 5.)],
                      options=dict(ftol=1e-12, gtol=1e-9))
    if not fitted.success:
        raise RuntimeError('Validation calibration failed: '+fitted.message)
    return dict(scale=float(fitted.x[0]), bias=float(fitted.x[1]),
                raw_logloss=objective([1., 0.])[0], fitted_logloss=float(fitted.fun))


def fit_validation(model, data, device):
    import torch
    assert data.split == 'val', 'Calibration must use validation data only'
    model.eval()
    values = []
    with torch.no_grad():
        for features, labels, domains in data.batches(32768):
            values.append(model.predict_native(features.to(device), domains.to(device)).cpu().numpy().reshape(-1))
    logits = np.concatenate(values)
    labels = np.asarray(data.label).reshape(-1)
    domains = np.asarray(data.domain).reshape(-1)
    parameters = [fit_affine(logits[domains==k], labels[domains==k]) for k in range(data.domains)]
    return dict(method='positive_affine_per_domain', fit_split='validation', rows=data.rows,
                parameters=parameters, scope='Only probabilities change; within-domain ranking is preserved.')


def apply_calibration(logits, domains, calibration):
    import torch
    assert calibration['fit_split'] == 'validation'
    parameters = calibration['parameters']
    assert all(p['scale'] > 0. for p in parameters)
    scale = torch.tensor([p['scale'] for p in parameters], device=logits.device, dtype=logits.dtype)
    bias = torch.tensor([p['bias'] for p in parameters], device=logits.device, dtype=logits.dtype)
    index = domains.long().reshape(-1)
    return logits*scale[index].reshape(-1, 1)+bias[index].reshape(-1, 1)
