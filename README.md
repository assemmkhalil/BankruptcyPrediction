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
The target column is “Class”, with 0 meaning not bankrupt and 1 meaning bankrupt.

**Original dataset:** Tomczak,Sebastian. (2016). Polish companies bankruptcy data. UCI Machine Learning Repository. https://doi.org/10.24432/C5F600

### Methodology:
1. Data Exploration: Explore the dataset to gain insights into its structure and characteristics, and detect data problems such as outliers, multicollinearity, class imbalances, etc.
2. Data Preprocessing: Handle predefined problems using various techniques to ensure data readiness for modeling.
3. Model Selection: Evaluate four tree-based and ensemble machine learning algorithms (Decision Tree, Random Forest, XGBoost, LightGBM) on different datasets created during the preprocessing stage. Select the most suitable training data and the champion model based on Recall score and AUC-PR, aiming to minimize False Negatives while also considering False Positives.
4. Feature Selection: Employ feature selection techniques such as PCA, RFE, feature importance with Random Forest, and feature permutation to identify the most relevant features for prediction.
5. Hyperparameter Tuning: Fine-tune the hyperparameters of the selected model using grid search and stratified k-fold cross-validation to optimize its performance.
