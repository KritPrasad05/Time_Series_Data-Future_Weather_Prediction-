<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Temperature Prediction Project</title>
</head>
<body>

<h1>🌡️ Temperature Prediction Project</h1>

<h2>Overview</h2>
<p>Predicting temperature fluctuations is a complex task influenced by environmental factors, pollution levels, and temporal trends. This project presents a comprehensive solution for temperature prediction using <strong>XGBoost Regressor</strong>.</p>

<hr>

<h2>📋 Table of Contents</h2>
<ul>
  <li><a href="#project-summary">Project Summary</a></li>
  <li><a href="#data-preparation">Data Preparation</a></li>
  <li><a href="#feature-engineering">Feature Engineering</a></li>
  <li><a href="#model-training">Model Training</a></li>
  <li><a href="#evaluation">Evaluation</a></li>
  <li><a href="#how-to-use">How to Use</a></li>
  <li><a href="#conclusion">Conclusion</a></li>
</ul>

<hr>

<h2 id="project-summary">📍 Project Summary</h2>
<p>This project builds a machine learning model to predict temperatures based on various factors:</p>
<ol>
  <li><strong>Data Exploration & Cleaning</strong>: Handling missing values and parsing date-time data.</li>
  <li><strong>Feature Engineering</strong>: Creating lagged values and extracting temporal features.</li>
  <li><strong>Model Training</strong>: Using <strong>XGBoost Regressor</strong> for efficient learning.</li>
  <li><strong>Evaluation</strong>: Custom scoring metric to measure prediction accuracy.</li>
</ol>

<hr>

<h2 id="data-preparation">🧹 Data Preparation</h2>

<h3>Import Libraries</h3>
<p>Essential libraries for data manipulation, visualization, and model training are imported at the start.</p>

<pre><code>import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
</code></pre>

<h3>Load the Dataset</h3>
<p>Load and inspect the dataset to identify missing values or inconsistencies.</p>

<pre><code>df = pd.read_csv("path_to_your_dataset.csv")</code></pre>

<h3>Data Cleaning</h3>
<ul>
  <li><strong>Fill Missing Values</strong>: Forward-fill (<code>ffill</code>) and backward-fill (<code>bfill</code>) to handle any gaps.</li>
  <li><strong>Datetime Parsing</strong>: Converting the <code>Datetime</code> column and creating additional time-based features for model training.</li>
</ul>

<pre><code>df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
df['Datetime'] = pd.to_datetime(df['Datetime'])
</code></pre>

<hr>

<h2 id="feature-engineering">⚙️ Feature Engineering</h2>
<p>To capture seasonal trends and temporal dynamics, the following features were engineered:</p>

<ul>
  <li><strong>Lag Features</strong>: Adding <code>lag1</code>, <code>lag2</code>, and <code>lag3</code> to incorporate recent historical values.</li>
  <li><strong>Temporal Features</strong>: Extracted additional attributes like <code>hour</code>, <code>day of the week</code>, <code>month</code>, <code>year</code>, etc., to account for cyclical patterns.</li>
</ul>

<h3>Final Feature Set</h3>

<pre><code>Features = ['Particulate_matter', 'SO2_concentration', 'O3_concentration', 
            'CO_concentration', 'NO2_concentration', 'Presure', 'Dew_point', 
            'Precipitation', 'Anonymous_X1', 'Wind_speed', 'Moisture_percent', 
            'lag1', 'lag2', 'lag3', 'hour', 'daysofweek', 'quater', 
            'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
Target = 'Temperature'
</code></pre>

<hr>

<h2 id="model-training">🏋️ Model Training</h2>

<h3>Initialize and Train XGBoost Model</h3>
<p>For high-performance learning, <strong>XGBoost Regressor</strong> was chosen. Its settings are customized to balance training speed and prediction accuracy.</p>

<pre><code>X_train = df[Features]
y_train = df[Target]

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', n_jobs=1, 
                       early_stopping_rounds=50, n_estimators=1000, 
                       max_depth=7, objective='reg:squarederror', 
                       learning_rate=0.01)

reg.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=100)
</code></pre>

<hr>

<h2 id="evaluation">📈 Evaluation</h2>

<p>A custom scoring metric was defined to measure model accuracy:</p>

<pre><code>score = max(0, 100 - mean_squared_error(y_test, y_pred))
</code></pre>

<h3>Model Performance</h3>
<p>Our model achieved a score of <strong>96.61</strong> on the test dataset, demonstrating its reliability and precision in predicting temperature.</p>

<hr>

<h2 id="how-to-use">💻 How to Use</h2>

<h3>Clone the Repository</h3>
<p>To run this project on your local machine, clone the repository with the following command:</p>

<pre><code>git clone https://github.com/yourusername/temperature-prediction.git
cd temperature-prediction
</code></pre>

<h3>Install Dependencies</h3>
<p>Ensure you have Python 3.x installed. Then, install the required packages with:</p>

<pre><code>pip install -r requirements.txt
</code></pre>

<h3>Running the Model</h3>
<p>After setting up the environment, you can load, preprocess, and train the model by running the provided notebook or script.</p>

<pre><code># Follow the steps in the notebook to preprocess the data, create features, and train the model.
</code></pre>

<hr>

<h2 id="conclusion">✨ Conclusion</h2>

<p>This project demonstrates a powerful approach for temperature prediction, using advanced feature engineering and robust machine learning techniques. The solution provides an adaptable foundation for future improvements and real-world applications.</p>

<p>Feel free to contribute to this repository or use the code for further research and exploration!</p>

</body>
</html>
