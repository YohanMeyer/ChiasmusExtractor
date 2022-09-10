# ChiasmusData

The repository containing all work concerning data retrieval and annotation for our masterthesis.

## Specifications for the annotation tool

### Goal : provide a tool for efficient annotation of chiasmi

#### First step : candidate extraction
__Input__ : Text containing chiasmi candidates \
__Input format__ : plain text in a .txt file \
__Output__ : The extracted chiasmi candidates \
__Output format__ : A .txt file with a list of blocks of 30 words, each containing an emphasized chiasmi candidate and its position in the original file, one block per line

#### Second step : candidate selection
__Input__ : chiasmi candidates \
__Input format__ : A .txt file with a list of blocks of 30 words, each containing an emphasized chiasmi candidate and its position in the original file, one block per line \
__Output__ : none if the candidate is rejected, the chiasmi and its context if it is accepted \
__Output format__ : same as the input

#### Third step : formatting
__Input__ : The original text and a list of chiasmi with their position in the text
__Input format__ : Two separate .txt files (one with the original raw text and one with the list of blocks of 30 words with specified chiasmi) \
__Output__ : A .xml file containing the annotated text \
__Output format__ : specified by Harris et al. (2018)








