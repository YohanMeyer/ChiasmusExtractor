# Chiasmus Detection Library

This is a library to detect chiasmi. It is based on the paper "Data-Driven Detection of General Chiasmi Using Lexical and Semantic Features".


Run `run.sh` to run the example script.
This script will download the required files and run the model.

## Prerequisites

* python3
* wget
* spacy de_core_news_lg
* see requirements.txt

## Data
* gerdracor/ contains a play from Schiller
* fasttext_models/ contains a model for word representations and sentence classification pre-trained on Wikipedia
* data_example/ contains an example used for demo

## Outputs
* candidates/
* processed/


## Citation

If you use this implementation in your work, please cite:

```
@inproceedings{schneider-etal-2021-data,
    title = "Data-Driven Detection of General Chiasmi Using Lexical and Semantic Features",
    author = {Schneider, Felix  and
      Barz, Bj{\"o}rn  and
      Brandes, Phillip  and
      Marshall, Sophie  and
      Denzler, Joachim},
    booktitle = "Proceedings of the 5th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic (online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.latechclfl-1.11",
    doi = "10.18653/v1/2021.latechclfl-1.11",
    pages = "96--100",
}
```