# Company Bankruptcy Prediction

### Overview:
The ability to predict a company's bankruptcy serves as a crucial early warning system for top management, enabling them to take proactive measures to prevent the company's financial downfall. Additionally, it provides valuable insights to investors by helping them make informed decisions by identifying companies to avoid for investment. Furthermore, for job seekers, it provides guidance by highlighting companies to avoid when seeking stable employment opportunities.
This project was done as a part of a [Kaggle competition](https://www.kaggle.com/competitions/debi-fintech-bankruptcy-prediction/overview) to address the challenge of predicting company bankruptcy by developing an effective machine learning model for financial risk assessment.

### Data:
The project was done using a subset of the Polish companies bankruptcy dataset. The data was collected from Emerging Markets Information Service (EMIS), which is a database containing information on emerging markets around the world. The bankrupt companies were analyzed in the period 2000-2012, while the still operating companies were evaluated from 2007 to 2013.
The dataset consists of 62 features, each representing a financial ratio such as:
- working capital / total assets
- book value of equity / total liabilities
- net profit / inventory
- EBITDA / total assets
- long-term liabilities / equity

The target column is “Class”, with 0 meaning not bankrupt and 1 meaning bankrupt. <br>

**Original dataset:** Tomczak,Sebastian. (2016). Polish companies bankruptcy data. UCI Machine Learning Repository. https://doi.org/10.24432/C5F600

### Methodology:
1. Data Exploration: Explored the dataset to gain insights into its structure and characteristics, and detect data problems such as nulls, duplicates, outliers, multicollinearity, class imbalances, etc.

2. Data Preprocessing: Handled problems detected in the data exploration step, e.g. removed duplicates, caped outliers, handled class imbalance with a combination of oversampling and undersampling, etc, to ensure data readiness for modeling.

3. Model Training and Selection: Evaluated four tree-based and ensemble algorithms (Decision Tree, Random Forest, XGBoost, LightGBM) on different versions of the dataset (with and without caping outliers with different IQR thresholds). Then selected the most suitable training data and the champion model with the main aim being to minimize False Negatives (failing to identify a company that will actually go bankrupt) while also considering False Positives, depending on Recall score (measures the proportion of actual positive cases that were correctly predicted) and AUC-PR (emphasizes the model's ability to capture positive cases while maintaining high precision).

4. Feature Selection: Employed feature selection techniques such as PCA, RFE, Feature Importance with Random Forest, and Feature Permutation to identify the most relevant features for prediction. Feature Importances with Random Forest achieved the highest Recall and AUC-PR score.

5. Hyperparameter Tuning: Fine-tuned the hyperparameters of the selected model using Grid Search and Stratified K-Fold Cross-Validation to optimize its performance.

### Results:
Champion Model: **LGBM using unfiltered data**
- Precision: Out of all the companies that the model predicted would go bankrupt, 66% actually did.
- Recall: Out of all the companies that actually did go bankrupt, the model predicted this this correctly for 70% of those companies.
- The AUC-PR score of 70% suggests that the model's ability to identify companies at risk of bankruptcy, considering precision and recall, is reasonably good but not perfect.
- The Macro F1 Score of 84% indicates that the model performs well in achieving a balance between precision and recall for predicting bankruptcy across multiple companies.

