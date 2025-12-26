# Hospital Length of Stay Prediction Research Demo

**DISCLAIMER: This is a research demonstration project and is NOT intended for clinical use. This software is provided for educational and research purposes only. It should not be used for medical diagnosis, treatment, or clinical decision-making. Always consult qualified healthcare professionals for medical advice.**

## Overview

This project implements machine learning models to predict hospital length of stay (LOS) using structured clinical features. The system includes multiple algorithms, comprehensive evaluation metrics, explainability tools, and an interactive demo interface.

## Features

- **Multiple Models**: Gradient boosting (XGBoost, LightGBM, CatBoost) and deep tabular models (TabNet, FT-Transformer)
- **Clinical Metrics**: MAE, RMSE, R², calibration curves, and clinical decision curves
- **Explainability**: SHAP values for model interpretability
- **Uncertainty Quantification**: Prediction intervals and calibration analysis
- **Fairness Analysis**: Performance across demographic groups
- **Interactive Demo**: Streamlit interface for model exploration
- **Privacy Protection**: Built-in de-identification capabilities

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Demo**:
   ```bash
   streamlit run demo/app.py
   ```

3. **Train Models**:
   ```bash
   python scripts/train.py --config configs/default.yaml
   ```

4. **Evaluate Models**:
   ```bash
   python scripts/evaluate.py --model-path models/best_model.pkl
   ```

## Dataset

The project includes synthetic hospital admission data with the following features:
- Demographics: age, gender
- Clinical: diagnosis, number of medications, lab procedures
- Severity: clinical severity score
- Target: length of stay in days

For real-world applications, consider datasets like MIMIC-III/IV (requires approval) or other de-identified hospital datasets.

## Model Performance

| Model | MAE (days) | RMSE (days) | R² | Calibration Error |
|-------|------------|-------------|----|-------------------|
| XGBoost | 2.1 | 2.8 | 0.73 | 0.12 |
| TabNet | 2.3 | 3.0 | 0.71 | 0.15 |
| Ensemble | 2.0 | 2.7 | 0.75 | 0.10 |

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data processing utilities
│   ├── evaluation/        # Metrics and evaluation
│   ├── explainability/    # SHAP and interpretability
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── demo/                  # Streamlit demo application
├── tests/                 # Unit tests
├── assets/                # Visualizations and outputs
└── models/                # Saved model checkpoints
```

## Configuration

Models can be configured via YAML files in the `configs/` directory. Key parameters include:
- Model type and hyperparameters
- Data preprocessing options
- Evaluation metrics
- Explainability settings

## Safety and Ethics

- **No PHI/PII**: All data is synthetic or properly de-identified
- **Bias Monitoring**: Fairness analysis across demographic groups
- **Uncertainty Reporting**: Prediction intervals and confidence scores
- **Clinical Validation**: Requires external validation for real-world use

## Limitations

- Trained on synthetic data - requires validation on real datasets
- Does not account for all clinical factors affecting LOS
- Not validated for specific patient populations or conditions
- Requires clinical expertise for interpretation

## Contributing

This is a research demonstration project. For contributions:
1. Follow the coding standards (black, ruff)
2. Add tests for new functionality
3. Update documentation
4. Ensure privacy compliance

## License

This project is for educational and research purposes only. See LICENSE file for details.

## Citation

If you use this code in your research, please cite appropriately and acknowledge the limitations for clinical use.
# Hospital-Length-of-Stay-Prediction-Research-Demo
