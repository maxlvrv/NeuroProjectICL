import numpy as np
import os
import json
from KeypointDetectorAndMatcher import Detector

tissueCyteFilePath = "./"
confocalFilePath = "./"
tissueCyteDescriptorFiles = "./TissueCyteDescriptorFiles/"
confocalFiles = []

question1 = input("Do you want to Match Images to Existing TissueCyte Data or Generate Features for New TissueCyte Data, (Match/Generate) ")

if (question1 == "Match"):
    question3 = input("Please enter the Name of the Folder where the Confocal Images are stored: ")
    confocalFilePath = confocalFilePath + question3
    try:
        tissueCyteFiles = os.listdir(tissueCyteDescriptorFiles)
        confocalFiles = os.listdir(confocalFilePath)
    except:
        print("You have entered an Incorrect Filename, Please restart the Program")
        exit()
    det = Detector()
    imageMatches = {}
    for x in confocalFiles:
        if x.endswith('.tif'):
            confocalDescriptor = det.computeKeypointsAndDescriptors(confocalFilePath + "/" + x, 20)
            currentHighest = 0
            currentBestImageMatch = "No Match"
            for y in tissueCyteFiles:
                try:
                    tissueCyteDescriptor = np.loadtxt(tissueCyteDescriptorFiles + y, dtype=np.float32)
                    numberOfKeypointMatches = det.matchImages(confocalDescriptor, tissueCyteDescriptor)
                    if (numberOfKeypointMatches > currentHighest):
                        currentHighest = numberOfKeypointMatches
                        currentBestImageMatch = y
                except:
                    print("Failed to Load image: " + y)
            imageMatches[x] = currentBestImageMatch
    print(imageMatches)
    
    #showMatches as an Image
    
    jsonMatching = json.dumps(imageMatches)
    f = open("ImageMatches.json", "w")
    f.write(jsonMatching)
    f.close()
    exit()
        
elif (question1 == "Generate"):
    question2 = input("Please enter the Name of the Folder where the TissueCyte data is stored: ")
    tissueCyteFilePath = tissueCyteFilePath + question2
    try:
       tissueCyteFiles = os.listdir(tissueCyteFilePath)
       print(tissueCyteFiles)
    except:
        print("You have entered an Incorrect Filename, Please restart the Program")
        exit()
    det = Detector()
    for x in tissueCyteFiles:
        if x.endswith('.tif'):
            descriptors = det.computeKeypointsAndDescriptors(tissueCyteFilePath + "/" + x, 50)
            np.savetxt((tissueCyteDescriptorFiles + x + ".txt"), descriptors)
    exit()
    
else:
    print("You have entered an Incorrect Command, Please restart the Program")
    exit()

'''
if mask:
        mask_path = '/mnt/TissueCyte80TB/181012_Gerald_KO/ko-Mosaic/SEGMENTATION_RES.tif'
        structure_list = 'LGv,LGd,RT,VENT'

if mask:
        #path = raw_input('NII/TIFF file path (drag-and-drop): ').rstrip()
        file, extension = os.path.splitext(mask_path)
        if extension == '.nii':
            seg = nib.load(mask_path).get_data()
        else:
            seg = io.imread(mask_path)
        print 'Loaded segmentation data'

ids = []
acr = []

if mask:
    anno_file = json.load(open('2017_annotation_structure_info.json'))
    structure_list = [x.strip() for x in structure_list.lower().split(",")]
    for elem in structure_list:
        a, i = get_structure(anno_file['children'], elem)[1:]
        ids.extend(i)
        acr.extend(a)

if mask:
            index = np.array([[],[],[]])
            if structure in seg:
                index = np.concatenate((index, np.array(np.nonzero(structure == seg))), axis=1)
            else:
                proceed = False

            if index.size > 0:
                zmin = int(index[0].min())
                zmax = int(index[0].max())
                ymin = int(index[1].min()*scale*downsize)
                ymax = int(index[1].max()*scale*downsize)
                xmin = int(index[2].min()*scale*downsize)
                xmax = int(index[2].max()*scale*downsize)
            else:
                proceed = False
        else:
            zmin = 0
            zmax = len(count_files)
            ymin = 0
            ymax = temp_size[1]*downsize
            xmin = 0
            xmax = temp_size[0]*downsize
'''

    
