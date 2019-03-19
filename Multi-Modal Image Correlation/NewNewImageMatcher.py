import numpy as np
import os
import json
from KeypointDetectorAndMatcher import Detector
from TissueCyteImagePreparation import ImageMasker

class ImageMatch():

    def __init__(self, descriptorFilePath):
        self.tissueCyteFilePath = "/mnt/TissueCyte80TB/181024_Gerald_HET/het-Mosaic/Ch2_Stitched_Sections"
        self.confocalFilePath = "/mnt/TissueCyte80TB/Marcus_ImageMatching/MatchedJSONLargerImages"#"/mnt/TissueCyte80TB/Marcus_ImageMatching/Confocal_Images"
        self.tissueCyteDescriptorFiles = descriptorFilePath #"/mnt/TissueCyte80TB/Marcus_ImageMatching/TissueCyteDescriptorFiles/"
        self.segmentationFilePath = '/mnt/TissueCyte80TB/181024_Gerald_HET/het-Mosaic/Het_seg_10um.tif'
        self.confocalFiles = []

    #question1 = input("Do you want to Match Images to Existing TissueCyte Data or Generate Features for New TissueCyte Data, (Match/Generate) ")

    def takeSecond(self, elem):
        return elem[1]

    def match(self, question1, fileName, setting, algorithm, crossCheck):

        if (question1 == "Match"):
            #question3 = input("Please enter the Name of the Folder where the Confocal Images are stored: ")
            #confocalFilePath = confocalFilePath + question3
            try:
                tissueCyteFiles = os.listdir(self.tissueCyteDescriptorFiles)
                self.confocalFiles = os.listdir(self.confocalFilePath)
            except:
                print("You have entered an Incorrect Filename, Please restart the Program")
                #exit()
            det = Detector(algorithm, crossCheck)
            imageMatches = {}
            for x in self.confocalFiles:
                if x.endswith('.tif'):
                    confocalDescriptor = det.computeKeypointsAndDescriptors(self.confocalFilePath + "/" + x, 20, False, setting)
                    #currentHighest = 0
                    #currentBestImageMatch = "No Match"
                    runningTally = []
                    for y in tissueCyteFiles:
                        try:
                            if (algorithm != "orb"):
                                tissueCyteDescriptor = np.loadtxt(self.tissueCyteDescriptorFiles + y, dtype=np.float32)
                            else:
                                tissueCyteDescriptor = np.loadtxt(self.tissueCyteDescriptorFiles + y, dtype=np.uint8)
                            if (algorithm == "sift"):
                                tissueCyteDescriptor = np.reshape(tissueCyteDescriptor,(-1,128))
                            elif (algorithm == "surf"):
                                tissueCyteDescriptor = np.reshape(tissueCyteDescriptor,(-1,64))
                            elif (algorithm == "orb"):
                                tissueCyteDescriptor = np.reshape(tissueCyteDescriptor,(-1,32))
                            if (algorithm != "orb"):
                                confocalDescriptor = np.float32(confocalDescriptor)
                            numberOfKeypointMatches = det.matchImages(confocalDescriptor, tissueCyteDescriptor)
                            runningTally.append((y, numberOfKeypointMatches))
                            #if (numberOfKeypointMatches > currentHighest):
                                #currentHighest = numberOfKeypointMatches
                                #currentBestImageMatch = (y,currentHighest)
                                #runningTally.append(currentBestImageMatch)
                        except:
                            print("Failed to Load image: " + y)
                    #imageMatches[x] = currentBestImageMatch
                    runningTally.sort(key=self.takeSecond)
                    imageMatches[x] = runningTally[-3:]
            print(imageMatches)
            
            #showMatches as an Image
            
            jsonMatching = json.dumps(imageMatches)
            f = open(fileName + ".json", "w")
            f.write(jsonMatching)
            f.close()
            #exit()
                
        elif (question1 == "Generate"):
            #question2 = input("Please enter the Name of the Folder where the TissueCyte data is stored: ")
            #tissueCyteFilePath = tissueCyteFilePath + question2
            masker = ImageMasker(self.segmentationFilePath)
            masker.analyseTissueCyteImages(self.tissueCyteFilePath, self.tissueCyteDescriptorFiles, setting, algorithm, crossCheck)
            #exit()
            
        else:
            print("You have entered an Incorrect Command, Please restart the Program")
            #exit()



