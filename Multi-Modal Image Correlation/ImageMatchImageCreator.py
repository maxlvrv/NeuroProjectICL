import json
import os
import cv2 as cv
from KeypointDetectorAndMatcher import Detector
from TissueCyteImagePreparation import ImageMasker
import numpy as np

class ImageMatchDraw():
    def __init__(self):
        self.confocalFilePath = "/mnt/TissueCyte80TB/Marcus_ImageMatching/Confocal_Images/"
        self.tissueCyteFilePath = "/mnt/TissueCyte80TB/181024_Gerald_HET/het-Mosaic/Ch2_Stitched_Sections"
        self.segmentationFilePath = "/mnt/TissueCyte80TB/181024_Gerald_HET/het-Mosaic/Het_seg_10um.tif"
        self.masker = ImageMasker(self.segmentationFilePath)

    def drawMatches(self, setting, algorithm, crossCheck, file):
        det = Detector(algorithm, crossCheck)

        with open(file, "r") as json_file:
            data = json.load(json_file)

        for x in data:
            if (x == "SOX14HET_131218_TiledIMGLeftThalamus_Slide18_Cropped.tif"):
                confocalDescriptor, confocalKeypoint, img1 = det.otherFunction(False, (self.confocalFilePath + x), 20, setting)  #
                for y in data[x]:
                    print(y)
                    tissueCyteDescriptor, tissueCyteKeypoint, img2 = self.masker.otherFunction(self.tissueCyteFilePath, setting, algorithm, crossCheck, y[0])   #
                    if (algorithm != "orb"):
                        confocalDescriptor = np.asarray(confocalDescriptor, np.float32)
                        tissueCyteDescriptor = np.asarray(tissueCyteDescriptor, np.float32)
                    else:
                        confocalDescriptor = np.asarray(confocalDescriptor, np.uint8)
                        tissueCyteDescriptor = np.asarray(tissueCyteDescriptor, np.uint8)
                    if (algorithm == "sift"):
                        tissueCyteDescriptor = np.reshape(tissueCyteDescriptor,(-1,128))
                        confocalDescriptor = np.reshape(confocalDescriptor, (-1,128))
                    elif (algorithm == "surf"):
                        tissueCyteDescriptor = np.reshape(tissueCyteDescriptor,(-1,64))
                        confocalDescriptor = np.reshape(confocalDescriptor, (-1,64))
                    elif (algorithm == "orb"):
                        tissueCyteDescriptor = np.reshape(tissueCyteDescriptor,(-1,32))
                        confocalDescriptor = np.reshape(confocalDescriptor, (-1,32))
                    matches = det.otherMatchingFunction(confocalDescriptor, tissueCyteDescriptor)
                    print(len(matches))
                    _, fileName = os.path.split(file)
                    if (crossCheck != True):
                        img3 = cv.drawMatchesKnn(img1, confocalKeypoint, img2, tissueCyteKeypoint, matches, None, flags=2)
                    else:
                        img3 = cv.drawMatches(img1, confocalKeypoint, img2, tissueCyteKeypoint, matches, None, flags=2)
                    cv.imwrite(('/mnt/TissueCyte80TB/Marcus_ImageMatching/DrawMatches/' + fileName + '_' + y[0] + '_' + x + '_' + 'MatchesImage.jpg'), img3)
