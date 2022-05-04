from preprocess import NB_Preprocessor
import os

import math

class NaiveBayesClassifier:

    def __init__(self, trainDataFileName, testFileName, parameterFileName, outputFileName):
        self.trainFileName = trainDataFileName
        self.parameterFileName = parameterFileName
        self.testFileName = testFileName
        self.outputFileName = outputFileName

            # variables for each class
        self.vocab = {}
        self.classPriors = {}
        self.totalTokensInClass = {}

            # variables used by every class
        self.totalLinesTrainingData = 0
        self.vocabSize = 0
        self.seenWords = {}
        self.commonWords = {}

# Methods used to train the model

    def addClass(self, className):
        self.classPriors[className] = 1
        self.totalTokensInClass[className] = 0
        self.vocab[className] = {}

    def trainClassifierOnVocabFile(self,vocabFileName ):

        with open(self.trainFileName,"r",encoding="utf8") as trainData:
            
            for line in trainData:      # every line in train data

                tokens = line.split()
                if(len(tokens) == 0):
                    continue

                self.totalLinesTrainingData+=1

                className = tokens[0]       # extract the labelled class name
                words = tokens[1:]      # extract the features 

                try:
                    self.classPriors[className] +=1     #increment the prior
                except KeyError:
                    self.addClass(className)        # or add the class to the list of classes

                    with open(vocabFileName,"r",encoding="utf8") as vocabFile:
                        for line in vocabFile:
                            tokens = line.split()
                            word = tokens[0]
                            self.vocab[className][word] = 0
                

                wordCounts = self.vocab[className]


                for i in range(0,len(words),2):     # training on the line of features/counts
                    w = words[i]
                    try:
                        count = int(words[i+1])
                    except ValueError:
                        print("Line#: " + str(self.totalLinesTrainingData))
                        print("Word# " + str(i))
                        print("PrevWord = " + w)
                        print("What should be a number: " + words[i+1])
                        return
                    
                    try:
                        wordCounts[w] +=count
                        self.totalTokensInClass[className] += count

                    except KeyError:
                        continue

        self.getPriorProbs()
        self.cleanseVocabList()

    def trainClassifierOnTrainingData(self ):

        with open(self.trainFileName,"r",encoding="utf8") as trainData:
            
            for line in trainData:      # every line in train data

                tokens = line.split()
                if(len(tokens) == 0):
                    continue

                self.totalLinesTrainingData+=1

                className = tokens[0]       # extract the labelled class name
                words = tokens[1:]      # extract the features 

                try:
                    self.classPriors[className] +=1     #increment the prior
                except KeyError:
                    self.addClass(className)        # or add the class to the list of classes
                

                wordCounts = self.vocab[className]

                for i in range(0,len(words),2):
                    w = words[i]
                    try:
                        count = int(words[i+1])
                    except ValueError:
                        print("Line#: " + str(self.totalLinesTrainingData))
                        print("Word# " + str(i))
                        print("PrevWord = " + w)
                        print("What should be a number: " + words[i+1])
                        return
                    
                    self.totalTokensInClass[className] += count
                    try:
                        wordCounts[w] +=count
                    except KeyError:
                        wordCounts[w] =count
                        try:
                            if (self.seenWords[w] == True):
                                pass 
                        except KeyError:
                            self.seenWords[w] = True
                            self.vocabSize+=1

        self.getPriorProbs()
   
    def getPriorProbs(self):
        for className in self.classPriors.keys():
            self.classPriors[className] /= self.totalLinesTrainingData


    def cleanseVocabList(self):

        classList = self.vocab.keys()

        for name in classList:
            self.vocabSize = len(self.vocab[name].keys())
            break


            
# Method used to write parameter file

    def writeParameters(self):
        with open(self.parameterFileName,"w",encoding="utf8") as file:
            file.write("VocabSize = " + str(self.vocabSize) + "\n")
            
            for className in self.classPriors.keys():
                file.write("*********************\n")
                file.write(str(className + " Prior = " + str(self.classPriors[className])  + "\n") )
                file.write(str(className + " NumTokens = " + str(self.totalTokensInClass[className])  + "\n") )
                file.write(str(className + " NumWords = " + str(len(self.vocab[className].keys()))  + "\n") )
            
                for word in self.vocab[className].keys():
                    if self.vocab[className][word] > 0:
                        file.write( str(word + " " + str(self.vocab[className][word])   + "\n") )
  

# Method used to obtain the common words

    def findCommonWords(self):
        with open("./movie-review-HW2/aclimdb/imdb.vocab") as vocab:
            for _ in range(50):
                line = vocab.readline()
                tokens = line.split()
                for t in tokens:
                    self.commonWords[t] = True


# Methods for testing from small dataset (different output vs big dataset)

    def testClassifierSmall(self):

        with open(self.outputFileName, "w", encoding="utf8" ) as outFile:
            with open(self.testFileName,"r",encoding="utf8") as testFile:
                    for line in testFile:
                        tokens = line.split()
                        words = tokens[1:]

                        probs = self.predictClassSmall(words)

                        for className in probs.keys():
                            outFile.write("P( " + className + " | d ) = " + str(probs[className]) + "\n")
            
            maxProb = -1
            maxClass = ""
            for className in probs.keys():
                if probs[className] > maxProb:
                    maxClass = className
                    maxProb = probs[className]

            outFile.write( "\nNaive Bayes will classify this document to be: " + maxClass )

    def predictClassSmall(self,words):

        classProbs = {}
        for className in self.classPriors.keys():
            prob = self.classPriors[className]
            wordCounts = self.vocab[className]
            denominator = self.totalTokensInClass[className] + self.vocabSize
            
            for i in range(0,len(words),2):
                w = words[i]
                count = int(words[i+1])


                try:
                    prob *=  count * (wordCounts[w] +1) / denominator 
                except KeyError:
                    prob *= (count/denominator)
                
            
            classProbs[className] = prob
            
        
        return classProbs        


# Testing from big dataset

    def testClassifier(self):

        totalLines = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        with open(self.outputFileName, "w", encoding="utf8" ) as outFile:
            with open(self.testFileName,"r",encoding="utf8") as testFile:

                    outFile.write( "Predicted Class " + " ---- Actual Class " + "\n")

                    for line in testFile:
                        totalLines +=1
                        tokens = line.split()
                        trueClass = tokens[0]
                        words = tokens[1:]

                        predictedClass = self.predictClass(words)

                        outFile.write(  predictedClass + " ---- " + trueClass + "\n" )


                        if predictedClass == trueClass:
                            if predictedClass == "pos":
                                TP+=1
                            else:
                                TN+=1
                        else:
                            if predictedClass == "pos":
                                FP+=1
                            else:
                                FN+=1
                                    
            outFile.write(f"\n\nConfusion Matrix:\n{TP}    {FP}       Pos Support:  " + str(TP+FP) + f"\n{FN}    {TN}     Neg Support: " + str(TN+FN))

            precision = round(TP/(TP+FP),3)
            recall = round(TP/(TP+FN),3)
            outFile.write(f"\n\nPrecision:  {TP}/({TP}+{FP}) = " + str(precision))
            outFile.write(f"\nRecall:  {TP}/({TP}+{FN}) = " + str(recall))
            outFile.write(f"\nF1-Score: 2*(" + str(round(precision*recall,3)) + ") /(" + str(round(precision+recall,3)) + ") = " + str(round(2*(precision*recall)/(precision+recall),3)))
            outFile.write("\n\nAccuracy: " + str(TP+TN) + "/" + str(totalLines) + " = " + str(round((TP+TN)/totalLines,3)))



    def predictClass(self,words):

        maxProb = None
        maxClass = None

        for className in self.classPriors.keys():
            prob = math.log2(self.classPriors[className])
            classMap = self.vocab[className]
            denominator = self.totalTokensInClass[className] + self.vocabSize


            for i in range(0,len(words),2):
                w = words[i]
                count = int(words[i+1])

                try:
                    if(self.commonWords[w]):
                        penalty = 0 # new feature, penalty for common words
                except KeyError:
                    penalty = 1 # if not in commonWords, no penalty.

                try:
                    prob += penalty * math.log2(  count*(classMap[w] +1) / denominator )
                except KeyError:
                    prob += math.log2( count/denominator)
            
            if maxProb==None or prob > maxProb:
                maxProb = prob
                maxClass = className
            
        
        return maxClass




# Methods called from main to run the program

def runOnSmallData():

    print("\nInput for small dataset will be read from this directory::  './small-dataset/  ")

    trainDir = "./small-dataset/train"
    testDir = "./small-dataset/test"

    trainFile = "./small-output/trainPreProcessed.txt"
    testFile =  "./small-output/testPreProcessed.txt"
    paramFile = "./small-output/small-BOW.NB.txt"
    outputFile = "./small-output/output.txt"

    try:
        os.mkdir("./small-output")
    except FileExistsError:
        pass

    NB_Preprocessor(trainDir,testDir,trainFile,testFile)
    print("Preprocessing done!")

    NB = NaiveBayesClassifier(trainFile, testFile, paramFile, outputFile)
    NB.trainClassifierOnTrainingData()
    NB.writeParameters()
    NB.testClassifierSmall()

    print("Outputs for small dataset written to this directory::   './small-output/' \n")    


def preProcessRealData():
    print("\nInput will be read for movie data from this directory:: './movie-review-HW2/aclImdb' ")

        #input directories
    trainDir = "./movie-review-HW2/aclImdb/train"
    testDir = "./movie-review-HW2/aclImdb/test"
    
        #output file names
    trainFile = "./movie-review-output/trainPreProcessed.txt"
    testFile = "./movie-review-output/testPreProcessed.txt"

    NB_Preprocessor(trainDir,testDir,trainFile,testFile)

    print("Preprocessing complete!")


def runOnRealData():

    trainFile = "./movie-review-output/trainPreProcessed.txt"
    testFile = "./movie-review-output/testPreProcessed.txt"
    paramFile = "./movie-review-output/movie-review-BOW.NB.txt"
    outputFile ="./movie-review-output/output.txt"
    vocabFile = "./movie-review-HW2/aclimdb/imdb.vocab"

    try:
        os.mkdir("./movie-review-output")
    except FileExistsError:
        pass


    if(os.path.exists(trainFile) and os.path.exists(testFile)):
        print("Preprocessing for movie-review dataset previously completed!")
    else:
        preProcessRealData()




    NB = NaiveBayesClassifier(trainFile, testFile, paramFile, outputFile)
    NB.trainClassifierOnVocabFile(vocabFile)
    NB.writeParameters()
    NB.findCommonWords()
    NB.testClassifier()

    print("Outputs written to this directory::   './movie-review-output/' ")    


if __name__ == "__main__":
    
    runOnSmallData()

    runOnRealData()
