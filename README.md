# CATALPA_EduNLP at PragTag-2023

This is the repository of our contribution to the 2023 PragTag challenge.

### Data

Place the challenge data (including auxiliary data) in the `./data` folder.

To obtain the train-test split we used in our experiments, run `spit_full_data.py`.

To look up gateway information for the auxiliary F1000Research data, fist execute `python3 data/extract_links.py` to crawl a list of the articles in the gateways of interest.
Then run `python3 data/analyze_domains.py` to build the index of which of the reviews in the auxiliary data are associated with which of the gateways.

### Experiments

#### 'No Data' Setting

For the similarity-based experiments, run `python3 experiment/no_data_similarity.py` to obtain results for both our internal data split as well as predictions on the challenge test data.
