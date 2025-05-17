# Mayoral Election Voter Turnout Probability Model

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)

## Overview

This repository contains a machine learning system for predicting voter turnout in mayoral elections. Using historical voting behavior extracted from New York State voter roll data, the model generates turnout probabilities that can be used to segment voters for campaign targeting. The system employs a LightGBM classifier with advanced behavioral pattern features to achieve high predictive accuracy.


## Features

- **Advanced Feature Engineering**: Extracts over 28 behavioral features from voter history
- **Temporal Pattern Detection**: Identifies changes in voting behavior over time
- **Cross-Election Analysis**: Analyzes participation across different election types
- **Campaign Segmentation**: Classifies voters into actionable targeting groups
- **Synthetic Data Generation**: Creates test data that preserves statistical properties
- **Privacy Preserving**: Works with anonymized data and can generate synthetic data

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mayoral-voter-prediction.git
cd mayoral-voter-prediction

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Feature Engineering

The `scripts/feature_eng.py` module is the core component of the system, transforming raw voter history strings into predictive features:

```python
from scripts.feature_eng import parse_voter_history, extract_clean_features_enhanced

# Load voter data
voters_df = pd.read_csv('data/raw/voter_file.csv')

# Define mayoral election dates (format: YYYYMMDD)
mayoral_dates = ['20211102', '20171107', '20131105', '20091103']

# Extract enhanced features
features_df, feature_cols = extract_clean_features_enhanced(
    voters_df, 
    parse_voter_history,
    mayoral_dates
)
```

### Feature Categories by Importance

1. **Temporal Pattern Features (35%)**: How voting behavior changes over time
2. **Behavior Change Features (28%)**: Detection of significant shifts in voting patterns
3. **Participation Rate Features (20%)**: Overall engagement in different election types
4. **Sequence Features (9%)**: Patterns in the timing and consistency of participation
5. **Diversity Features (4%)**: Engagement across different election types
6. **Basic Features (3%)**: Simple counts and binary indicators
7. **Demographic Features (2%)**: Age, registration date, etc.

## Voter Segmentation

The `scripts/segment.py` module classifies voters into actionable campaign targeting groups:

```python
from scripts.segment import segment_voters_for_campaign

# Segment voters based on turnout probabilities
segmented_df = segment_voters_for_campaign(predictions_df)

# Analyze segment distribution
segment_counts = segmented_df['segment'].value_counts()
print(segment_counts)
```

### Segmentation Categories

- **HIGH_PROPENSITY (85%+ probability)**: Reliable voters needing voter protection
- **LIKELY (65-85% probability)**: Strong voters targeted for GOTV efforts
- **PERSUADABLE (35-65% probability)**: Key targets for persuasion campaigns (Canvassing)
- **LOW_PROPENSITY (<35% probability)**: Low-engagement voters receiving minimal outreach

## Synthetic Data Generation

For testing and development, the `scripts/fake_data.py` module can generate realistic synthetic voter data:

```python
from scripts.fake_data import create_synthetic_sample_similar_to_real

# Generate 5,000 synthetic voter records
synthetic_df = create_synthetic_sample_similar_to_real(
    'sample_data.csv', 
    column_names, 
    5000, 
    "synthetic_voter_data.csv"
)
```

## Notebooks

### Training Pipeline (notebook/training.ipynb)

This notebook demonstrates the end-to-end training process:
- Data loading and preprocessing
- Feature extraction
- Optimal model selection
- Model training (LightGBM)
- Probability calibration
- Model evaluation 


### Use Case for 2025 (notebook/use_case.ipynb)

Shows practical application for an upcoming election:
- Generating predictions for 2025 mayoral election using synthesized data
- Segmenting voters into targeting groups
- Creating campaign targeting lists

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib
- seaborn
- Faker (for synthetic data generation)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research or project, please cite:

```
@software{mayoral-voter-prediction,
  author = {Nicholas Gutin},
  title = {Mayoral Election Voter Turnout Probability Model},
  year = {2025},
  url = {https://github.com/yourusername/mayoral-voter-prediction}
}
```

## Acknowledgments

- New York State Board of Elections for the voter file format documentation
- Contributors to the open-source libraries used in this project
