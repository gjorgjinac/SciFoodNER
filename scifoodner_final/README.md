### SciFoodNER

SciFoodNER is a food Named Entity Recognition (NER) and Named Entity Linking (NEL) model for scientific text.
The models can recognize food entities from raw text, as well as link them to the Hansard Taxonomy, the FoodOn Ontology, and the Systematised Nomenclature of Medicine Clinical Terms (SNOMEDCT). Each task is treated separately, i.e. different models are trained for NER and each of the NEL tasks.
It performs fine-tuning of BERT, BioBERT, RoBERTa and SciBERT models on a corpus of 500 abstracts of scientific articles annotated for the existence of food entities and their ids in the Hansard Taxonomy, the FoodOn Ontology and SNOMEDCT.

The project is organized as follows:
- The required libraries that need to be installed are listed in requirements.txt
- The NER_data directory contains the full datasets for each of the tasks, stored in IOB format
- The main.py scipt is the main entrypoint for splitting the data into folds and running the models. The folds are stored in the directory 'folds_{number_of_folds}', while the results are stored in the directory 'results_{number_of_folds}'
- The notebooks are used purely for analyzing the results and are not required for training the models.