# ChiasmusExtractor

The repository containing all work concerning data retrieval and annotation for our Master's Theses.

## Global specifications for the extraction pipeline

### Goal : provide a pipeline for efficient extraction and annotation of chiasmi (and antimetabole)

#### First step : candidate extraction
__Input__ : Text containing chiasmi candidates \
__Input format__ : plain text in a .txt file \
__Output__ : The extracted chiasmi candidates \
__Output format__ : A .jsonl file with one candidate per line, its context and position in the original .txt file

#### Second step : candidate selection
__Input__ : chiasmi candidates \
__Input format__ : A .jsonl file with one candidate per line, its context and position in the original .txt file \
__Output__ : The accepted chiasmi
__Output format__ : A .jsonl file with one chiasmus per line, its context and position in the original .txt file

#### Third step : formatting
__Input__ : The original text and a list of chiasmi \
__Input format__ : The original .txt file and the .jsonl file (output from the previous step) \
__Output__ : A .xml file containing the annotated text \
__Output format__ : specified by Harris et al. (2018)


## Candidate extraction tool

The antimetabole candidate extraction is based on lemmas, as the tool searches for inversions of lemmas in the "AB(C...C)BA" form in a window of thirty tokens. Unfortunately, the extraction of chiasmi candidates based on word embedding is not providing satisfactory results. \
The window ignores punctuation and stopwords.

To run it in the `src/` folder :
```shell
python3 candidates-extractor.py [input.txt]
```

## Annotation tool

### How to use Doccano

We decided to integrate the free and open-source software Doccano to facilitate the manual annotation process.
For more details, please refer to their GitHub or documentation page : 
- https://github.com/doccano/doccano
- https://doccano.github.io/doccano/

## Output formatting

To transform the output from Doccano to a more universal scheme inspired from Harris et al. (2018), run this in the `src/` folder :
```shell
python3 annotation-formatter.py [annotated-input.jsonl input.txt]
```

## Citing our Master's Theses

Guillaume Berthomet:
```
@mastersthesis{berthomet2023_antimetabole,
  author={Guillaume Berthomet},
  title={Detecting Salient Antimetaboles in English Texts using Deep and Transfer Learning},
  school={University of Passau \& INSA Lyon},
  year={2023}
}
```

Yohan Meyer:
```
@mastersthesis{meyer2023_chiasmi,
  author={Yohan Meyer},
  title={Application of comprehensive and responsible Machine Learning methods to the detection of chiasmi's rhetorical salience},
  school={University of Passau \& INSA Lyon},
  year={2023}
}
```
