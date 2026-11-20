"""Calibration behavior and validation-only boundary checks."""
import unittest
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from train_script.hdcs_calibration import fit_affine, apply_calibration, fit_validation


class CalibrationChecks(unittest.TestCase):
    def test_calibration_improves_probability_loss_and_preserves_ranking(self):
        logits = np.linspace(-5., 1., 4000)
        labels = (np.random.RandomState(2022).random_sample(4000) < expit(1.5*logits-.6)).astype(float)
        fitted = fit_affine(logits, labels)
        self.assertGreater(fitted['scale'], 0.)
        self.assertLess(fitted['fitted_logloss'], fitted['raw_logloss']-.005)
        calibration = dict(fit_split='validation', parameters=[fitted, fitted])
        values = apply_calibration(torch.tensor(logits).reshape(-1, 1),
                                   torch.arange(4000).remainder(2).reshape(-1, 1), calibration).numpy().reshape(-1)
        self.assertAlmostEqual(roc_auc_score(labels, logits), roc_auc_score(labels, values), places=12)

    def test_calibration_cannot_fit_test_data(self):
        class TestData:
            split = 'test'
        with self.assertRaisesRegex(AssertionError, 'validation'):
            fit_validation(None, TestData(), 'cpu')


if __name__ == '__main__':
    unittest.main()
