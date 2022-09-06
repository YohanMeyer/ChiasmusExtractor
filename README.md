# ChiasmusData

The repository containing all work concerning data retrieval and annotation for our masterthesis.

## Specifications for the annotation tool

### Goal : provide a tool for efficient annotation of chiasmi

#### First step : candidate selection
__Input__ : Text containing chiasmi candidates \
__Input format__ : plain text in a .txt file \
__Output__ : The extracted chiasmi candidates \
__Output format__ : blocks of 30 words containing each an emphasized chiasmi candidate or a .txt file with a list of said blocks, one per line

#### Second step : processing
__Input__ : chiasmi candidates \
__Input format__ : blocks of 30 words containing an emphasized chiasmi candidate or a .txt file with a list of said blocks, one per line \
__Output__ : none if the candidate is rejected, the chiasmi and its context if it is accepted \
__Output format__ : same as the input

#### Third step : formatting
__Input__ : Text containing a specified chiasmus \
__Input format__ : a block of text of unspecified length containing a specified chiasmus or two separate .txt files (one with raw text and one with the list of blocks of 30 words with specified chiasmi) \
__Output__ : The same text under XML-annotation \
__Output format__ : specified by Harris et al. (2018)








