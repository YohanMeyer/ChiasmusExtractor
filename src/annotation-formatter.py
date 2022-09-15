import os.path
import sys
import jsonlines
import xml.etree.ElementTree as ET
from xml.dom import minidom
from operator import itemgetter

from utility import *

# -- Initializing the project, getting file contents --

if __name__ == '__main__' and len(sys.argv) == 3:
    annotatedFileName = sys.argv[1]
    rawFileName = sys.argv[2]
else:
    if __name__ == '__main__' and len(sys.argv) == 2:
        fileName = sys.argv[1]
        annotatedFileName = fileName + '-annotated.jsonl'
        rawFileName = fileName + '.txt'
    else:
        rawFileName = input('Enter the name of the original file : ')
        annotatedFileName = input('Enter the name of the annotated file : ')

annotatedJson = []
with jsonlines.open(os.path.join("..", "annotated", annotatedFileName)) as reader:
    for lineDict in reader:
        annotatedJson.append(lineDict)

rawContent = get_file_content(rawFileName, "inputs")
if(rawContent == -1):
    exit(0)

figureName = input("Please enter the name of the annotated rhetoric figure : ").lower() or "chiasmus"

print("figure :", figureName)

# -- Tidying up the annotated file --

annotatedTrueJson = [dict for dict in annotatedJson if len(dict["cats"]) == 1 and dict["cats"][0] == "TrueChiasmus"]
annotatedTrueJson = sorted(annotatedTrueJson, key=lambda d: (d["startBlock"] + d['entities'][0][0], d['entities'][1][0]))


annotatedXMLFileName = os.path.join("..", "annotated", os.path.splitext(os.path.basename(annotatedFileName))[0] + ".xml")

# initializing variables to loop over the document's figures
textIndex = 0
figureIndex = 0
document = ET.Element("document")
newXMLFigure = None

for trueChiasmus in annotatedTrueJson:
    
    startBlock = trueChiasmus["startBlock"]
    startChiasmus = startBlock + trueChiasmus["entities"][0][0]
    if(textIndex == 0 and figureIndex == 0):
        document.text = rawContent[0:startChiasmus]
        textIndex = startChiasmus
    else:
        newXMLFigure.tail = rawContent[textIndex:startChiasmus]
        
    newXMLFigure = ET.SubElement(document, figureName + "-" + str(figureIndex))
    
    # initializing variables to loop over the figure's terms
    newXMLTerm = None
    figureTextIndex = 0
    
    for termIndex, term in enumerate(trueChiasmus["entities"]):
        if(newXMLTerm is not None):
            newXMLTerm.tail = rawContent[(startBlock + figureTextIndex) : (startBlock + term[0])]
        newXMLTerm = ET.SubElement(newXMLFigure, figureName + "-" + term[2])
        # figure out a way to get the whole word
        newXMLTerm.text = rawContent[(startBlock + term[0]):(startBlock + term[1])]
        
        figureTextIndex = term[1]
        
    textIndex = startBlock + trueChiasmus["entities"][-1][1]
    figureIndex += 1
    
newXMLFigure.tail = rawContent[textIndex:]
    
with open(annotatedXMLFileName, 'w') as fileOut:
    xmlString = minidom.parseString(ET.tostring(document)).toprettyxml(indent="    ")
    fileOut.write(xmlString)
    fileOut.close()
    
print("\n---------")
print("Annotated XML file stored in", annotatedXMLFileName)