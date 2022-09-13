import itertools
import copy
import os.path
import sys
import json

from utility import *

# -- Initializing the project --

if __name__ == '__main__' and len(sys.argv) == 3:
    annotatedFileName = sys.argv[1]
    rawFileName = sys.argv[2]
else:
    annotatedFileName = input('Enter the name of the annotated file to convert : ')
    rawFileName = input('Enter the name of the raw file which was annotated : ')

annotatedContent = get_file_content(annotatedFileName, "../outputs/")
if(annotatedContent == -1):
    exit(0)
rawContent = get_file_content(rawFileName, "../inputs/")
if(rawContent == -1):
    exit(0)

print(annotatedContent)
print(rawContent)