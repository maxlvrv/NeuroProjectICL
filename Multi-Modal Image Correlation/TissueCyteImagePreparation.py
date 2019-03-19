import numpy as np
import cv2 as cv
import os
from PIL import Image
from skimage import io
from natsort import natsorted
import nibabel as nib
from KeypointDetectorAndMatcher import Detector
import warnings, json

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

class ImageMasker():

    def __init__(self, tissueCyte_path):
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
        warnings.simplefilter('ignore', Image.DecompressionBombWarning)
        Image.MAX_IMAGE_PIXELS = 1000000000
        self.mask_path = tissueCyte_path
        self.structure_list = 'LGd-sh'
        self.seg = None
        self.ids = None
        self.acr = None

    def getSegmentationData(self):
        file, extension = os.path.splitext(self.mask_path)
        if extension == '.nii':
            seg = nib.load(self.mask_path).get_data()
        else:
            seg = io.imread(self.mask_path)
        print('Loaded Segmentation data')
        return seg

    def get_children(self, json_obj, acr, ids):
        for obj in json_obj:
            if obj['children'] == []:
                acr.append(obj['acronym'])
                ids.append(obj['id'])
            else:
                acr.append(obj['acronym'])
                ids.append(obj['id'])
                self.get_children(obj['children'], acr, ids)
        return (acr, ids)

    def get_structure(self, json_obj, acronym):
        found = (False, None)
        for obj in json_obj:
            if obj['acronym'].lower() == acronym:
                #print obj['acronym'], obj['id']
                [acr, ids] = self.get_children(obj['children'], [], [])
                #print ids
                if ids == []:
                    acr = [obj['acronym']]
                    ids = [obj['id']]
                    return (True, acr, ids)
                else:
                    acr.append(obj['acronym'])
                    ids.append(obj['id'])
                    return (True, acr, ids)
            else:
                found = self.get_structure(obj['children'], acronym)
                if found:
                    return found

    def loadAnnotationData(self):
        ids = []
        acr = []
        anno_file = json.load(open('2017_annotation_structure_info.json'))
        self.structure_list = [x.strip() for x in self.structure_list.lower().split(",")]
        for elem in self.structure_list:
            a, i = self.get_structure(anno_file['children'], elem)[1:]
            ids.extend(i)
            acr.extend(a)
        print('Feature Matching in Structures: '+str(acr))
        return ids, acr

    def analyseTissueCyteImages(self, count_path, descriptor_path, setting, algorithm, crossCheck):
        count_files = []
        count_files += [each for each in os.listdir(count_path) if each.endswith('.tif')]
        count_files = natsorted(count_files)
        temp = Image.open(count_path+'/'+count_files[0])
        temp_size = temp.size
        temp = None
        structure_index = 0
        ids, acr = self.loadAnnotationData()
        seg = self.getSegmentationData()
        det = Detector(algorithm, crossCheck)
        for name, structure in zip(acr,ids):
            index = np.array([[],[],[]])
            if structure in seg:
                index = np.concatenate((index, np.array(np.nonzero(structure == seg))), axis=1)

            if index.size > 0:
                zmin = int(index[0].min())
                zmax = int(index[0].max())

            print(str(zmin) + "/" + str(zmax))

            for slice_number in range(zmin,zmax):
                # Load image and convert to dtype=float and scale to full 255 range
                image = Image.open(count_path+'/'+count_files[slice_number])
                image = np.array(image).astype(float)
                image = np.multiply(np.divide(image,np.max(image)), 255.)

                newImg = self.applyImageMask(image, seg, structure, slice_number, temp_size)

                #newImg = cv.cvtColor(np.array(newImg), cv.COLOR_RGB2BGR)
                newImg = np.uint8(newImg)

                descriptors = det.computeKeypointsAndDescriptors(newImg, 50, True, setting)
                try:
                    if (descriptors != None):
                        np.savetxt((descriptor_path + str(name) + "_" + count_files[slice_number] + ".txt"), descriptors)
                except:
                    np.savetxt((descriptor_path + str(name) + "_" + count_files[slice_number] + ".txt"), descriptors)
                print(str(name) + " " + str(slice_number) + " Completed")
                    

    def applyImageMask(self, image, seg, structure, slice_number, temp_size):
        mask_image = np.array(Image.fromarray(seg[slice_number]).resize(tuple([int(x) for x in temp_size]), Image.NEAREST))
        mask_image[mask_image!=structure] = 0
        image[mask_image==0] = 0
        
        # Crop image with idx
        mask_image = image>0
        idx = np.ix_(mask_image.any(1),mask_image.any(0))
        mask_image = None
        row_idx = idx[0].flatten()
        col_idx = idx[1].flatten()
        image = image[idx]

        return image

    def otherFunction(self, count_path, setting, algorithm, crossCheck, x):
        count_files = []
        count_files += [each for each in os.listdir(count_path) if each.endswith('.tif')]
        count_files = natsorted(count_files)
        temp = Image.open(count_path+'/'+count_files[0])
        temp_size = temp.size
        temp = None
        structure_index = 0
        try:
            if (self.ids == None and self.acr == None):
                self.ids, self.acr = self.loadAnnotationData()
                ids = self.ids
                acr = self.acr
            else:
                ids = self.ids
                acr = self.acr
        except:
            ids = self.ids
            acr = self.acr
        try:
            if (self.seg == None):
                self.seg = self.getSegmentationData()
                seg = self.seg
            else:
                seg = self.seg
        except:
            seg = self.seg
        det = Detector(algorithm, crossCheck)
        #for x in drawList:
        descriptors = None
        keypoints = None
        newImg = None
        for name, structure in zip(acr,ids):
            print(x)
            x, ext = os.path.splitext(x)
            slice_number = x.split("_")
            slice_number = slice_number[2]
            slice_number = int(slice_number[1] + slice_number[2] + slice_number[3])
            
            image = Image.open(count_path+'/'+count_files[slice_number-1])
            image = np.array(image).astype(float)
            image = np.multiply(np.divide(image,np.max(image)), 255.)

            newImg = self.applyImageMask(image, seg, structure, slice_number, temp_size)

                #newImg = cv.cvtColor(np.array(newImg), cv.COLOR_RGB2BGR)
            newImg = np.uint8(newImg)

            descriptors, keypoints, newImg = det.otherFunction(True, newImg, 50, setting)
            print(str(slice_number) + " Completed")
            self.structure_list = 'LGd-sh'
            print(len(descriptors))
            return descriptors, keypoints, newImg

