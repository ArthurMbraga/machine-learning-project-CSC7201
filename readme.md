# Machine Learning Project

This project contains various scripts and notebooks for data preprocessing, feature engineering, model training, and evaluation. The data required for this project is already included in the `project/data` folder.

## Project Structure

- `data/`: Directory containing the datasets used in the project.
- `benchmark.ipynb`: Jupyter notebook for benchmarking different models.
- `data_visualization.ipynb`: Jupyter notebook for visualizing the data.
- `dataset.py`: Script for loading and preprocessing the dataset.
- `feature_engineering.py`: Script for feature engineering.
- `project.ipynb`: Main Jupyter notebook for running the project.

## Instructions

1. **Data Preprocessing**: The data preprocessing steps are implemented in `dataset.py`. This script loads the data from the `data` folder and performs necessary preprocessing steps.

2. **Feature Engineering**: The feature engineering steps are implemented in `feature_engineering.py`. This script generates new features from the existing data.

3. **Model Training and Evaluation**: The main notebook for running the project is `project.ipynb`. This notebook includes code for training and evaluating different machine learning models.

4. **Data Visualization**: The `data_visualization.ipynb` notebook contains various visualizations to understand the data better.

5. **Benchmarking**: The `benchmark.ipynb` notebook is used for benchmarking different models to compare their performance.

## Running the Project

To run the project, follow these steps:

1. Open `project.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the cells in the notebook sequentially to preprocess the data, perform feature engineering, train the models, and evaluate their performance.

## Dependencies

Make sure you have the following dependencies installed:

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- missingno

You can install the dependencies using the following command:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn missingno
```