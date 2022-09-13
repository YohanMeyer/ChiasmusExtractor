# ChiasmusData

The repository containing all work concerning data retrieval and annotation for our masterthesis.

## Global specifications for the annotation tool

### Goal : provide a tool for efficient annotation of chiasmi

[TODO] update

#### First step : candidate extraction
__Input__ : Text containing chiasmi candidates \
__Input format__ : plain text in a .txt file \
__Output__ : The extracted chiasmi candidates \
__Output format__ : A .txt file with a list of blocks of 30 words (and adding 25 subsequent characters for context) on a line, and the position of a chiasmi candidate in the original file on the next line

#### Second step : candidate selection
__Input__ : chiasmi candidates \
__Input format__ : A .txt file with a list of blocks of 30 words (and adding 25 subsequent characters for context) on a line, and the position of a chiasmi candidate in the original file on the next line \
__Output__ : none if the candidate is rejected, the chiasmi and its context if it is accepted \
__Output format__ : same as the input

#### Third step : formatting
__Input__ : The original text and a list of chiasmi with their position in the text
__Input format__ : Two separate .txt files (one with the original raw text and one with the list of blocks of 30 words with specified chiasmi) \
__Output__ : A .xml file containing the annotated text \
__Output format__ : specified by Harris et al. (2018)


## Candidate extraction tool

[TODO] A few words about its functioning + how to run it

## Annotation tool

### How to use Doccano

We decided to integrate the free and open-source software Doccano to facilitate the manual annotation process.
For more details, please refer to their GitHub or documentation page : 
- https://github.com/doccano/doccano
- https://doccano.github.io/doccano/

#### How to run it locally

[TODO]

#### With Docker

Please note that having Docker installed is a prerequisite.
[TODO]







