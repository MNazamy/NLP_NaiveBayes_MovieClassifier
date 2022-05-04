import os
import time

class NB_Preprocessor:

    
    def __init__(self, trainDirectory, testDirectory,  outputTrainFile,outputTestFile):

        start = time.time()
        self.preProcessAllFiles(trainDirectory, outputTrainFile)
        end= time.time()
        print("Finished train preprocessing in " +str(end-start) + " seconds" )

        start = time.time()
        self.preProcessAllFiles(testDirectory,outputTestFile)     
        end= time.time()
        print("Finished test preprocessing in " +str(end-start) + " seconds" )

    def cleanseToken(self, t):
        t.replace('.','')
        t.replace(',','')
        t.replace('*','')
        t.replace('<','')
        t.replace('>','')
        t.replace('@','')
        t.replace('#','')
        t.replace('$','')
        t.replace('^','')
        t.replace('&','')
        t.replace('*','')
        t.replace('@','')
        t.replace( '/' , '')
        t.replace( '\\' , '')

        return t

    def preProcessAllFiles(self,rootDir, outputFile):

        with open(outputFile,"w",encoding="utf8") as out:

            for subdir, dirs, files in os.walk(rootDir):


                className = os.path.basename(os.path.normpath(subdir))


                for file in files:
                    filePath = os.path.join(subdir,file)

                    with open(filePath,"r",encoding="utf8") as f:
                        
                        thisCount = {}
                        cleansedTokens = []

                        for line in f:

                            tokens = line.split()

                            for t in tokens:
                                t = t.lower()

                                if t.isalpha() :
                                    cleansedTokens.append(t)
                                

                                else:
                                    
                                    t = self.cleanseToken(t)

                                    if(len(t) == 1):
                                        cleansedTokens.append(t)

                                    elif(len(t) == 0):
                                        continue
                                    
                                    else:

                                        prevStop = 0
                                        prevIndex = 0
                                        nextIndex=1

                                        while nextIndex <= len(t):
                                            while nextIndex < len(t) and t[prevIndex].isalpha() == t[nextIndex].isalpha():
                                                prevIndex+=1
                                                nextIndex+=1
                                            
                                            newToken = t[prevStop:nextIndex]
                                            cleansedTokens.append(newToken)

                                            prevStop = nextIndex
                                            prevIndex = nextIndex
                                            nextIndex +=1
                                        
                        
                            for t in cleansedTokens:
                                
                                try:
                                    thisCount[t] +=1
                                except KeyError:
                                    thisCount[t] = 1
                    
                            writeLine = className + " "
                            for word in thisCount.keys():
                                writeLine += (word +  " " + str(thisCount[word]) + " " ) 
                            writeLine += "\n"
                            out.write(writeLine)


                    

