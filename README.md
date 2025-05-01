# 🫀 Heart Disease Prediction Model

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0.2-orange)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3.0-red)](https://pandas.pydata.org/)

A machine learning model that predicts the presence of heart disease based on various medical parameters. This project demonstrates the implementation of a binary classification model using Logistic Regression and includes comprehensive model evaluation and visualization.

## 📋 Table of Contents
- [Features](#-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Results Visualization](#-results-visualization)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features
- Binary classification using Logistic Regression
- Feature importance analysis and visualization
- Comprehensive model evaluation metrics
- Automated results storage and tracking
- Data preprocessing and scaling
- Confusion matrix visualization
- Detailed performance reports

## 📊 Dataset
The model uses the Heart Disease dataset which includes the following parameters:
- Age
- Sex
- Chest pain type (4 values)
- Resting blood pressure
- Serum cholesterol in mg/dl
- Fasting blood sugar > 120 mg/dl
- Resting electrocardiographic results
- Maximum heart rate achieved
- Exercise induced angina
- ST depression induced by exercise
- Slope of the peak exercise ST segment
- Number of major vessels colored by fluoroscopy
- Thalassemia

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Run the model:
```bash
python model.py
```

2. Check the results in the `results/` directory:
- Model performance metrics (`model_metrics_[timestamp].txt`)
- Confusion matrix visualization (`confusion_matrix_[timestamp].png`)
- Feature importance plot (`feature_importance_[timestamp].png`)

## 📁 Project Structure
```
heart-disease-prediction/
├── model.py              # Main model implementation
├── requirements.txt      # Project dependencies
├── README.md            # Project documentation
└── results/             # Directory containing model outputs
    ├── model_metrics_*.txt
    ├── confusion_matrix_*.png
    └── feature_importance_*.png
```

## 📈 Model Performance
The model's performance metrics are stored in timestamped files within the `results/` directory and include:
- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix details
- Feature importance analysis

## 📊 Results Visualization
The model generates several visualizations:
1. **Confusion Matrix**: Shows the model's prediction accuracy
2. **Feature Importance Plot**: Displays the relative importance of each feature
3. **Performance Metrics**: Detailed text report of model performance

## 🔮 Future Improvements
- [ ] Implement cross-validation
- [ ] Add more advanced models (Random Forest, XGBoost)
- [ ] Feature engineering and selection
- [ ] Hyperparameter tuning
- [ ] Model interpretation techniques (SHAP values)
- [ ] Web interface for predictions
- [ ] API endpoint for model serving

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors
- Hemaizi syrine - Initial work

## 🙏 Acknowledgments
- UCI Machine Learning Repository for the dataset
- Scikit-learn team for the machine learning tools
- All contributors who have helped improve this project 
