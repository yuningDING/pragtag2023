# CATALPA_EduNLP at PragTag-2023

This is the repository of our contribution to the 2023 PragTag challenge.

### Data

Place the challenge data (including auxiliary data) in the `./data` folder.

To obtain the train-test split we used in our experiments, run `spit_full_data.py`.

To look up gateway information for the auxiliary F1000Research data, first, execute `python3 data/extract_links.py` to crawl a list of the articles in the gateways of interest.
Then run `python3 data/analyze_domains.py` to build the index of which of the reviews in the auxiliary data are associated with which of the gateways.

The arr.json data is transformed from the original ARR data with the script in ./auxiliary_data/data_transformation.py, mapping [paper summary] to [Recap], [summary of strengths] to [Strength], [summary of weaknesses] to [Weakness] and [comments, suggestions and typos] to [Todo]. The arr_sampled.csv data is the sampled variant of arr.json according to the class distribution in the full-data.

The keep_words_* data are used in + Word Normalization setting.

### Experiments

#### 'Full-data' Setting

Run run_full_data.sh

#### 'Low Data' Setting

To run the BERT-based model, run run_low_data.sh.
To run the SBERT-based model, splitting the data once, execute `python3 experiment/low_data_similarity.py`.
To train four separate models on different splits of the training data, run `python3 experiment/low_data_similarity_folds.py`.

#### 'No Data' Setting

For the SBERT-based experiments, run `python3 experiment/no_data_similarity.py` to obtain results for both our internal data split and predictions on the challenge test data.
