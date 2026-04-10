# Code Review: Pass 11 — Model Registry Integration

**Date**: 2026-04-10
**Reviewer**: Claude Code

## Stats

- Files Modified: 5
- Files Added: 3 (autogluon_pyfunc.py, test_model_registry.py, task file)
- Files Deleted: 0
- New lines: ~170
- Deleted lines: ~5

## Issues

```
severity: medium
file: src/windcast/training/backends.py
line: 65
issue: Double .to_pandas() conversion in log_model — signature + input_example
detail: Both XGBoostBackend.log_model() (line 65 + 71) and AutoGluonBackend.log_model() 
  (line 149 + 157) call X_val.to_pandas() twice — once for infer_signature() and once for 
  input_example. For large validation sets, the first .to_pandas() is expensive (full 
  conversion). The input_example uses .head(5) first which is fine, but the signature 
  inference converts the entire X_val to pandas.
suggestion: Convert once and reuse: `X_pd = X_val.to_pandas()`, then 
  `infer_signature(X_pd, y_pred)` and `input_example=X_pd.head(5)`. This halves the 
  pandas conversion overhead.
```

```
severity: low
file: tests/training/test_model_registry.py
line: 11
issue: Unused import: pytest
detail: pytest is imported at line 11 but only @pytest.fixture uses it. Ruff doesn't 
  flag it because pytest fixtures are special-cased. However, the import IS needed at 
  runtime for the fixture decorator, so this is a non-issue. No action needed.
suggestion: None — false positive on review, import is required.
```

```
severity: low
file: src/windcast/models/autogluon_pyfunc.py
line: 30
issue: .values on AutoGluon predict() may return ExtensionArray in edge cases
detail: The line `self.predictor.predict(model_input).values` assumes AutoGluon returns 
  a pandas Series. This is correct for standard TabularPredictor usage. The type: ignore 
  comment is appropriate. No real risk in practice.
suggestion: None — current code is correct for the documented AutoGluon API.
```

## Summary

Code review passed. No critical or high-severity issues detected.

The implementation is clean, well-structured, and follows existing codebase patterns:
- Protocol extension follows the existing `log_child_artifacts` pattern
- Lazy imports follow the existing `mlflow.xgboost` import pattern
- Tests mirror the existing `test_harness.py` fixture and assertion patterns
- MLflow 3.x API usage is correct (name= instead of artifact_path=, model_uri return)
- Registration logic is minimal and well-guarded (3 conditions: register_model_name + log_models + best_model_uri)

The one actionable optimization is the double `.to_pandas()` call in both backends (medium severity — performance, not correctness).
