# Personalized survival probabilities of SARS-CoV-2 positive patients by explainable machine learning

Code repository for the paper "[Personalized survival probabilities of SARS-CoV-2 positive patients by explainable machine learning](https://doi.org/10.1038/s41598-022-17953-y)" by Zucco et al., 2022.

## Software requirements
Tested on Ubuntu 20.04 (Windows Subsystem for Linux 2)

### 1. Clone this repository
Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to run:
```bash
git clone https://github.com/PERSIMUNE/COVIMUN_DT.git
```

### 2. Create an environment
Install [conda](https://github.com/PERSIMUNE/COVIMUN_DT.git) to create a new python environment.

```
  conda env create -f covimun_ml_env.yml
  conda activate covimun_ml_env
  pip install -r requirements.txt
```

Alternatively:
- A virtual environment can be created using `virtualenv` 
- The `pip install -r requirements.txt` can be executed in your base environment given that python 3.9 is already installed in your system.

## Run predictions

Data in `.csv`/`.xlsx` format needs to be provided with values for one patient in each row. The columns and values are described below:


| Column name              | Description                                                    | Value                                       |
|--------------------------|----------------------------------------------------------------|---------------------------------------------|
| index                    | Index given by the user                                        | String or numerical                         |
| age                      | Age                                                            | Last value observed                         |
| num_ordmeds              | Number of ordered medicines                                    | Count in the last year                      |
| in_hosp                  | Admitted at the time of first positive test                    | 1 (yes) / 0 (no)                            |
| BMI                      | Body Mass Index                                                | Last value in the last month                |
| diagnoses_count_DZ01     | Encounter for other special examination (Z01), Diagnose, count | Count in the last 3 years                   |
| pandemic_weeks           | Pandemic week                                                  | Weeks since 1st or March 2020               |
| sex_at_birth             | Sex                                                            | 0 (female) / 1 (male)                       |
| ordmed_count_A06AD       | Laxatives (A06AD), Ordered Medicine, count                     | Count in the last year                      |
| num_diagnoses            | Number of diagnoses                                            | Count in the last 3 years                   |
| ordmed_count_N02BE       | Paracetamol (N02BE), Ordered Medicine, count                   | Count in the last year                      |
| laboratory_lastvalue_LYM | Absolute Lymphocyte count (LYM), laboratory, last value        | Latest value in the last month (G/L)        |
| ordmed_count_C03CA       | Loop diuretics (C03CA), Ordered Medicine, count                | Count in the last year                      |
| days_in_hospital         | Cumulative days in hospital within the last 3 years            | Sum of days in hospital in the last 3 years |
| ordmed_count_N01AH       | Opioid anesthetics (N01AH), Ordered Medicine, count            | Count in the last year                      |
| HOSP_BEFORE              | Previous admissions in the last 3 years                        | Count in the last 3 years                   |
| diagnoses_count_DG30     | Alzheimer's disease (G30), Diagnose, count                     | Count in the last 3 years                   |
| diagnoses_count_DZ03     | Encounter for medical observation (Z03), Diagnose, count       | Count in the last 3 years                   |
| diagnoses_count_DI10     | Essential hypertension (I10), Diagnose, count                  | Count in the last 3 years                   |
| adminmed_count_C03CA     | Loop diuretics (C03CA), Administered medicine, count           | Count in the last year                      |
| ordmed_count_A11EA       | Vitamin B-complex (A11EA), Ordered Medicine, count             | Count in the last year                      |
| adminmed_count_A12AX     | Calcium + vitamin D  (A12AX),   Administered medicine, count   | Count in the last year                      |

An example can be seen at `data_to_predict.csv`, this file can be edited or a different file can be provided with the flag `--input_data`. To run predictions using the default examples use:

```
python ./run_predictions.py
```
For custom files an options please follow the help which can be accessed by `python ./run_predictions.py -h`

```
usage: run_predictions.py [-h] [--input_data INPUT_DATA] [--models_folder MODELS_FOLDER] [--output_folder OUTPUT_FOLDER] [--col_info COL_INFO] [--plot_curves {True,False}] [--explain {True,False}]

Predict survival probabilities withing 12 weeks from a first SARS-CoV-2 positive test

optional arguments:
  -h, --help            show this help message and exit
  --input_data INPUT_DATA
                        Data frame with feature names as header and one patient per row (default: ./data_to_predict.csv)
  --models_folder MODELS_FOLDER
                        Path to the folder with the trained models (default: ./trained_models)
  --output_folder OUTPUT_FOLDER
                        Path to the folder in which results will be saved (default: ./results)
  --col_info COL_INFO   Path to the file with the columns metadata (default: ./metadata/column_details.csv)
  --plot_curves {True,False}
                        Generate cumulative incidence and instant hazard plots for each patient (default: True)
  --explain {True,False}
                        Generate individual explanations for each patient (default: True)
```

## How to cite

```
Zucco, A. G., Agius, R., Svanberg, R., Moestrup, K. S., Marandi, R. Z., MacPherson, C. R., Lundgren, J., Ostrowski, S. R., & Niemann, C. U. (2022). 
Personalized survival probabilities for SARS-CoV-2 positive patients by explainable machine learning. 
Scientific Reports, 12(1), 13879. https://doi.org/10.1038/s41598-022-17953-y
```

BibTex format:

```

@article{zucco_personalized_2022,
	title = {Personalized survival probabilities for {SARS}-{CoV}-2 positive patients by explainable machine learning},
	volume = {12},
	copyright = {2022 The Author(s)},
	issn = {2045-2322},
	url = {http://www.nature.com/articles/s41598-022-17953-y},
	doi = {10.1038/s41598-022-17953-y},
	abstract = {Interpretable risk assessment of SARS-CoV-2 positive patients can aid clinicians to implement precision medicine. Here we trained a machine learning model to predict mortality within 12 weeks of a first positive SARS-CoV-2 test. By leveraging data on 33,938 confirmed SARS-CoV-2 cases in eastern Denmark, we considered 2723 variables extracted from electronic health records (EHR) including demographics, diagnoses, medications, laboratory test results and vital parameters. A discrete-time framework for survival modelling enabled us to predict personalized survival curves and explain individual risk factors. Performance on the test set was measured with a weighted concordance index of 0.95 and an area under the curve for precision-recall of 0.71. Age, sex, number of medications, previous hospitalizations and lymphocyte counts were identified as top mortality risk factors. Our explainable survival model developed on EHR data also revealed temporal dynamics of the 22 selected risk factors. Upon further validation, this model may allow direct reporting of personalized survival probabilities in routine care.},
	language = {en},
	number = {1},
	urldate = {2022-09-04},
	journal = {Scientific Reports},
	author = {Zucco, Adrian G. and Agius, Rudi and Svanberg, Rebecka and Moestrup, Kasper S. and Marandi, Ramtin Z. and MacPherson, Cameron Ross and Lundgren, Jens and Ostrowski, Sisse R. and Niemann, Carsten U.},
	month = aug,
	year = {2022},
	keywords = {Machine learning, Prognosis, Viral infection},
	pages = {13879},
}

```
