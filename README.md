# Fairness in Predictive Process Monitoring: Pre-processing Effects on Bias

This repository contains the implementation and analysis for the master's thesis project investigating how preprocessing decisions affect fairness in Predictive Process Monitoring (PPM) systems.

## 📋 Project Overview

This study examines the impact of preprocessing choices, specifically prefix length and encoding strategies, on fairness outcomes in PPM systems. The research addresses the critical question: **How do preprocessing steps, particularly prefix length and encoding type, affect fairness in predictive process monitoring?**

## 🎯 Research Question

Do decisions during the data preparation phase, particularly prefix length and encoding type, affect fairness in predictive process monitoring?

## 📊 Dataset

- **Dataset**: `hiring_log_medium.csv` (simulated hiring process data)
- **Source**: Pohl et al. (2023) - A Collection of Simulated Event Logs for Fairness Assessment in Process Mining
- **Size**: 69,055 rows, 10,000 unique cases
- **Activities**: 12 unique activities
- **Resources**: 50 unique resources

## 🔧 Implementation

### Preprocessing Configurations

The study tests 6 different configurations:

| Prefix Length | Encoding Type | Description |
|---------------|---------------|-------------|
| 3 | Simple | First 3 activities (activity only) |
| 3 | Complex | First 3 activities + resource info |
| 5 | Simple | First 5 activities (activity only) |
| 5 | Complex | First 5 activities + resource info |
| 10 | Simple | First 10 activities (activity only) |
| 10 | Complex | First 10 activities + resource info |

### Key Files

- `01_build_aequitas_inputs.py` - Main preprocessing and model training
- `02A_make_aequitas_csvs.py` - Prepare data for Aequitas
- `02B_run_aequitas_audit.py` - Run fairness audit
- `03_summarize_reports.py` - Generate summary reports
- `03b_key_findings.py` - Extract key findings
- `create_clean_visualizations.py` - Generate result visualizations

## 📈 Results

### Main Findings

1. **Prefix Length Effect**: Shorter prefixes (3 activities) show lower bias levels than longer prefixes (10 activities)
2. **Encoding Strategy Impact**: Simple encoding produces less bias than complex encoding
3. **Configuration Interaction**: Best fairness performance: Prefix 3 + Simple encoding
4. **Attribute Sensitivity**: Sensitive attributes show consistent bias patterns, while dynamic attributes vary by configuration

### Fairness Metrics

- **PPR (Predicted Positive Rate)**: Positive prediction rate for each group
- **FDR (False Discovery Rate)**: FP/(TP+FP)
- **FOR (False Omission Rate)**: FN/(TN+FN)
- **TPR (True Positive Rate)**: TP/(TP+FN)
- **FPR (False Positive Rate)**: FP/(FP+TN)

### 80% Rule Violations

- **Threshold**: Disparity < 0.8 or > 1.25 indicates fairness violation
- **Severity Levels**:
  - Moderate: 0.8 ≤ disparity ≤ 1.25
  - High: 0.6 ≤ disparity < 0.8 or 1.25 < disparity ≤ 1.67
  - Severe: disparity < 0.6 or disparity > 1.67

## 🛠️ Technical Details

### Environment
- Python 3.8+
- PyCharm IDE (recommended)
- Aequitas toolkit for fairness assessment

### Dependencies
```
pandas
numpy
scikit-learn
matplotlib
seaborn
aequitas
```

### Installation
```bash
git clone https://github.com/yourusername/fairness-ppm.git
cd fairness-ppm
pip install -r requirements.txt
```

### Usage
```bash
# Run complete pipeline
python 01_build_aequitas_inputs.py
python 02A_make_aequitas_csvs.py
python 02B_run_aequitas_audit.py
python 03_summarize_reports.py
python create_clean_visualizations.py
```

## 📁 Directory Structure

```
├── hiring_log_medium.csv          # Main dataset
├── 01_build_aequitas_inputs.py    # Preprocessing and model training
├── 02A_make_aequitas_csvs.py      # Aequitas input preparation
├── 02B_run_aequitas_audit.py      # Fairness audit execution
├── 03_summarize_reports.py        # Report generation
├── 03b_key_findings.py            # Key findings extraction
├── create_clean_visualizations.py # Visualization generation
├── outputs/                       # Model outputs
├── outputs_aeq/                   # Aequitas formatted data
├── outputs_reports/               # Fairness reports
└── README.md                      # This file
```

## 📊 Output Files

### Generated Reports
- `_summary_all_disparities.csv` - Complete fairness analysis
- `_violations_only.csv` - Only fairness violations
- `prefix{length}_{encoding}_aeq_disparities.csv` - Individual configuration results

### Visualizations
- `ppr_violations_heatmap.png` - PPR violations by configuration
- `average_disparity_heatmap.png` - Average disparity by configuration
- `attribute_type_violations.png` - Sensitive vs dynamic attribute violations
- `severity_distribution.png` - Violation severity distribution
- `worst_disparities.png` - Top 10 worst disparities
- `metric_violations.png` - Violations by metric type
- `sensitive_attributes_boxplot.png` - Sensitive attributes analysis
- `configuration_performance.png` - Overall configuration performance

## 🔬 Methodology

### Data Processing
1. **Prefix Extraction**: Extract sequences of specified length from process traces
2. **Encoding**: Apply simple (activity-only) or complex (activity+resource) encoding
3. **Label Generation**: Binary classification (Hire = 1, Not Hire = 0)

### Model Training
- **Algorithm**: Decision Tree Classifier
- **Split**: 80/20 train-test split (case-based)
- **Prediction**: Binary output (0 or 1)

### Fairness Assessment
- **Toolkit**: Aequitas
- **Reference Groups**: 
  - Gender: Most common value
  - Age: 45-64 (largest group)
  - Citizenship: T (True)
  - German Speaking: T (True)
  - Religious: Most common value
- **Statistical Significance**: n≥30 filter applied

## 📚 References

- Mehrabi, N., et al. (2021). A Survey on Bias and Fairness in Machine Learning
- Pohl, T., et al. (2023). A Collection of Simulated Event Logs for Fairness Assessment in Process Mining
- Saleiro, P., et al. (2018). Aequitas: A Bias and Fairness Audit Toolkit

## 👨‍🎓 Author

This project was developed as part of a master's thesis at Humboldt University Berlin.

## 📄 License

This project is for academic research purposes.

## 🤝 Contributing

This is an academic research project. For questions or suggestions, please contact the author.

---

**Note**: This implementation addresses the research question of how preprocessing decisions impact fairness in PPM systems, providing empirical evidence that technical choices have ethical implications.
