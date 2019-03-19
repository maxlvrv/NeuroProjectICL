import numpy as np
import cv2 as cv
import scipy.ndimage as ndi
from scipy.ndimage._ni_support import _normalize_sequence
#from filters.rollingballfilt import rolling_ball_filter

class Detector():

    def __init__(self, detector, crossCheck):
        if (detector == "sift"):
            self.sift = cv.xfeatures2d.SIFT_create()
            if (crossCheck == True):
                self.bf = cv.BFMatcher(crossCheck=True)
            else:
                self.bf = cv.BFMatcher()
        elif (detector == "surf"):
            self.sift = cv.xfeatures2d.SURF_create()
            if (crossCheck == True):
                self.bf = cv.BFMatcher(crossCheck=True)
            else:
                self.bf = cv.BFMatcher()
        elif (detector == "orb"):
            self.sift = cv.ORB_create()
            if (crossCheck == True):
                self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            else:
                self.bf = cv.BFMatcher(cv.NORM_HAMMING)
            
        self.detector = detector
        self.crossCheck = crossCheck

    def computeKeypointsAndDescriptors(self, img, scale, isTissueCyte, setting):
        try:
            if (isTissueCyte == False):
                image = cv.imread(img)
                image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
                image = self.normalisePixelValues(image)
                if (setting == 1):
                    image = self.applyGaussianFilter(image)
                elif (setting == 2):
                    image = self.applyGaussianFilter(image)
                    if (image.shape[0]*image.shape[1] > (12*2)**2 and np.max(image) != 0.):
                        image, background = self.rolling_ball_filter(np.uint8(image), 24)
                elif (setting == 3):
                    image = self.applyGaussianFilter(image)
                    image = self.changeScale(image, scale)
                elif (setting == 4):
                    if (image.shape[0]*image.shape[1] > (12*2)**2 and np.max(image) != 0.):
                        image, background = self.rolling_ball_filter(np.uint8(image), 24)
                    
            else:
                if (setting == 1):
                    image = self.applyGaussianFilter(img)
                elif (setting == 2):
                    image = self.applyGaussianFilter(img)
                    if (image.shape[0]*image.shape[1] > (12*2)**2 and np.max(image) != 0.):
                        image, background = self.rolling_ball_filter(image, 24)
                elif (setting == 3):
                    image = self.applyGaussianFilter(img)
                    image = self.changeScale(image, scale)
                elif (setting == 4):
                    if (img.shape[0]*img.shape[1] > (12*2)**2 and np.max(img) != 0.):
                        image, background = self.rolling_ball_filter(img, 24)
                    else:
                        image = img
                elif (setting == 0):
                    image = img
                elif (setting == 5):
                    image = self.applyGaussianFilter(img)
                    if (image.shape[0]*image.shape[1] > (12*2)**2 and np.max(image) != 0.):
                        image, background = self.rolling_ball_filter(image, 24)
            try:
                gray= cv.cvtColor(image,cv.COLOR_BGR2GRAY)
            except:
                gray = image
            print(gray.shape)
            #if (self.detector == "sift" || self.detector == "surf"):
            kp, des = self.sift.detectAndCompute(gray,None)
            #elif (self.detector == "orb"):
                #kp = self.sift.detect(gray, None)
                #kp, des = self.sift.compute(gray, kp)
            return des
        except:
            print("Zero Error")
            if (self.detector == "sift"):
                return np.zeros((1,128))
            elif (self.detector == "surf"):
                return np.zeros((1,64))
            elif (self.detector == "orb"):
                return np.zeros((1,32))
            
    def matchImages(self, descriptor1, descriptor2):
        if (self.crossCheck == False):
            matches = self.bf.knnMatch(descriptor1, descriptor2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                   good.append([m])
            return len(good)
        else:
            matches = self.bf.match(descriptor1, descriptor2)
            return len(matches)

    def applyGaussianFilter(self, img):
        blurredImg = cv.GaussianBlur(img,(5,5),0)
        return blurredImg

    def normalisePixelValues(self, img):
        normal = cv.normalize(img, 0, 255, norm_type=cv.NORM_MINMAX)
        return normal

    '''def showMatches(self, img1, img2, kp1, kp2, matchSet):
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matchSet,flags=2)
        plt.imshow(img3),plt.show()'''

    def changeScale(self, img, scale):
        width = int((img.shape[1] * scale)/100)
        height = int((img.shape[0] * scale)/100)
        dim = (width, height)
        resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
        return resized

    def rolling_ball_filter(self, data, ball_radius, spacing=None, top=False, **kwargs):
        data = data.astype(np.int16)
        ndim = data.ndim
        if spacing is None:
            spacing = _normalize_sequence(1, ndim)
        else:
            spacing = _normalize_sequence(spacing, ndim)

        radius = np.asarray(_normalize_sequence(ball_radius, ndim))
        mesh = np.array(np.meshgrid(*[np.arange(-r, r + s, s) for r, s in zip(radius, spacing)], indexing="ij"))
        structure = np.uint8(2 * np.sqrt(np.absolute(2 - ((mesh / radius.reshape(-1, *((1,) * ndim)))**2).sum(0))))
        structure[~np.isfinite(structure)] = 0

        if not top:
            # ndi.white_tophat(y, structure=structure, output=background)
            background = cv.erode(data, structure, **kwargs)
            background = cv.dilate(background, structure, **kwargs)
        else:
            # ndi.black_tophat(y, structure=structure, output=background)
            background = cv.dilate(data, structure, **kwargs)
            background = cv.erode(background, structure, **kwargs)

        data_corr = data - background
        data_corr[data_corr<0] = 0

        return data_corr.astype(np.uint8), background.astype(np.uint8)

    def otherFunction(self, isTissueCyte, img, scale, setting):
        try:
            if (isTissueCyte == False):
                image = cv.imread(img)
                image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
                image = self.normalisePixelValues(image)
                if (setting == 1):
                    image = self.applyGaussianFilter(image)
                elif (setting == 2):
                    image = self.applyGaussianFilter(image)
                    if (image.shape[0]*image.shape[1] > (12*2)**2 and np.max(image) != 0.):
                        image, background = self.rolling_ball_filter(np.uint8(image), 24)
                elif (setting == 3):
                    image = self.applyGaussianFilter(image)
                    image = self.changeScale(image, scale)
                elif (setting == 4):
                    if (image.shape[0]*image.shape[1] > (12*2)**2 and np.max(image) != 0.):
                        image, background = self.rolling_ball_filter(np.uint8(image), 24)
                    
            else:
                if (setting == 1):
                    image = self.applyGaussianFilter(img)
                elif (setting == 2):
                    image = self.applyGaussianFilter(img)
                    if (image.shape[0]*image.shape[1] > (12*2)**2 and np.max(image) != 0.):
                        image, background = self.rolling_ball_filter(image, 24)
                elif (setting == 3):
                    image = self.applyGaussianFilter(img)
                    image = self.changeScale(image, scale)
                elif (setting == 4):
                    if (img.shape[0]*img.shape[1] > (12*2)**2 and np.max(img) != 0.):
                        image, background = self.rolling_ball_filter(img, 24)
                    else:
                        image = img
                elif (setting == 0):
                    image = img
                elif (setting == 5):
                    image = self.applyGaussianFilter(img)
                    if (image.shape[0]*image.shape[1] > (12*2)**2 and np.max(image) != 0.):
                        image, background = self.rolling_ball_filter(image, 24)
            try:
                gray= cv.cvtColor(image,cv.COLOR_BGR2GRAY)
            except:
                gray = image
            print(gray.shape)
            #if (self.detector == "sift" || self.detector == "surf"):
            kp, des = self.sift.detectAndCompute(gray,None)
            #elif (self.detector == "orb"):
                #kp = self.sift.detect(gray, None)
                #kp, des = self.sift.compute(gray, kp)
            return des, kp, gray
        except:
            print("Zero Error")
            if (self.detector == "sift"):
                return np.zeros((1,128))
            elif (self.detector == "surf"):
                return np.zeros((1,64))
            elif (self.detector == "orb"):
                return np.zeros((1,32))

    def otherMatchingFunction(self, descriptor1, descriptor2):
        if (self.crossCheck == False):
            matches = self.bf.knnMatch(descriptor1, descriptor2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                   good.append([m])
            return good
        else:
            matches = self.bf.match(descriptor1, descriptor2)
            return matches


#image, background = rolling_ball_filter(np.uint8(image), 8)
