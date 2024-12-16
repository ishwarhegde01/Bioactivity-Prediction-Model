# Bioactivity Prediction of Trypsin-I Using PubChem Fingerprinting Descriptors

This repository contains the implementation of a machine learning model for predicting the bioactivity of compounds targeting Trypsin-I (ChEMBL ID: CHEMBL209). The project involves descriptor calculation, exploratory data analysis, model building, and a web application for deployment. 

---

## Project Structure

- **`ML_drug_discovery_1.ipynb`**: Data loading and preprocessing. The dataset is imported from the `data` folder and prepared for further analysis.
- **`ML_drug_discovery_2.ipynb`**: Exploratory Data Analysis (EDA) and calculation of Lipinski descriptors to assess drug-likeness.
- **`ML_drug_discovery_3.ipynb`**: Descriptor calculation using PaDEL-Descriptor software. The focus is on computing PubChem fingerprinting descriptors. The PaDEL-Descriptor software is downloaded using `wget` in this notebook.
- **`ML_drug_discovery_4.ipynb`**: Machine learning model building for bioactivity prediction of Trypsin-I using the Random Forest algorithm and the processed descriptors.
- **`app.py`**: A Streamlit-based web application for demonstrating the predictive model.

---

## Requirements

To set up and run the project, ensure the following dependencies are installed:

### Python Libraries

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `streamlit`
- Any other libraries mentioned in the notebooks

### External Tools

- [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) for descriptor calculations.

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required Python libraries and dependencies using the `env.yml` file:
   ```bash
   conda env create -f env.yml
   conda activate <environment_name>
   ```

3. Run the Jupyter notebooks in sequence (`ML_drug_discovery_1.ipynb` to `ML_drug_discovery_4.ipynb`) for the complete pipeline.

4. Launch the web application:
   ```bash
   streamlit run app.py
   ```

---

## Detailed Workflow

### 1. Data Preparation
In the first notebook, the dataset is imported from the `data` folder, and a dataframe is created for training. Preprocessing steps are performed to clean and organize the data efficiently.

### 2. Exploratory Data Analysis
In the second notebook, various visualizations and calculations are performed to evaluate the dataset. Lipinski descriptors are calculated to analyze drug-likeness properties of the compounds.

### 3. Descriptor Calculation
Using the PaDEL-Descriptor software, PubChem fingerprinting descriptors are calculated. The software is downloaded using `wget` within this notebook to streamline the process. This step ensures that the features used for training are both informative and relevant for the prediction task.

### 4. Model Building
In the fourth notebook, the Random Forest algorithm is used to train a machine learning model on the computed descriptors. Various evaluation metrics are employed to determine the model's performance in predicting the bioactivity of Trypsin-I.

### 5. Web Application
The `app.py` script contains a Streamlit-based web application that allows users to interact with the predictive model. Users can input compound descriptors and obtain bioactivity predictions in real-time.

---

## Future Work

- Incorporate additional descriptors and feature selection methods.
- Extend the application to other biological targets.
- Implement advanced machine learning algorithms and hyperparameter optimization.

---

## References

- [ChEMBL Database](https://www.ebi.ac.uk/chembl/)
- [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/)
- [Lipinski's Rule of Five](https://en.wikipedia.org/wiki/Lipinski%27s_rule_of_five)

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

Special thanks to the developers of PaDEL-Descriptor and the ChEMBL database for providing essential tools and resources for this research.
