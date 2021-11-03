# Personalized survival probabilities of SARS-CoV-2 positive patients by explainable machine learning

Official code repository for the paper "Personalized survival probabilities of SARS-CoV-2 positive patients by explainable machine learning" by Zucco et al.

## Software requirements
Tested on Ubuntu 20.04, run the following commands.

```
  conda env create -f covimun_ml_env.yml
  conda activate covimun_ml_env
  pip install -r requirements.txt
```

## Run predictions

```
python ./run_predictions.py
```

```
usage: run_predictions.py [-h] [--input_data INPUT_DATA] [--models_folder MODELS_FOLDER] [--output_folder OUTPUT_FOLDER] [--col_info COL_INFO] [--plot_curves {True,False}] [--explain {True,False}]

Predict survival probabilites withing 12 weeks from a first SARS-CoV-2 positive test

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
Zucco, A. G., Agius, R., Svanberg, R., Moestrup, K. S., Marandi, R. Z., MacPherson, C. R., Lundgren, J., Ostrowski, S. R., & Niemann, C. U. (2021). 
Personalized survival probabilities for SARS-CoV-2 positive patients by explainable machine learning (p. 2021.10.28.21265598). 
https://doi.org/10.1101/2021.10.28.21265598
```

BibTex format

```
@techreport{zuccoPersonalizedSurvivalProbabilities2021,
  title = {Personalized Survival Probabilities for {{SARS}}-{{CoV}}-2 Positive Patients by Explainable Machine Learning},
  author = {Zucco, Adrian G. and Agius, Rudi and Svanberg, Rebecka and Moestrup, Kasper S. and Marandi, Ramtin Z. and MacPherson, Cameron Ross and Lundgren, Jens and Ostrowski, Sisse R. and Niemann, Carsten U.},
  year = {2021},
  month = oct,
  pages = {2021.10.28.21265598},
  institution = {{Cold Spring Harbor Laboratory Press}},
  doi = {10.1101/2021.10.28.21265598},
  langid = {english}
}
```