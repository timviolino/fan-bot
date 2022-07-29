import os
import cv2 as cv
import math
from pytesseract import pytesseract
import numpy as np
from PIL import Image
from fuzzywuzzy import fuzz

GRAPH_FILE = "graph300.png"
PATH_TO_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class fanBot:
    def __init__(self):
        self.tr = self.textReader(PATH_TO_TESSERACT)
        self.pe = self.physicsEngine()
        self.ip = self.imageProcessor()
        self.dp = self.dataProcessor()
        self.s = self.speech()
        self.fc = self.graph()

    def acquirefc(self):
        fileType = ''
        while(fileType not in self.ip.imgTypes):
            file = self.s.ask("Fan Curve")
            fileType = os.path.splitext(file)[1]
        self.ip.setImage(file)

    def setAxes(self):
        self.fc.size = self.ip.size
        try:
            axisTextImgs = self.ip.cropTextFromImage()  # returns cropped images with text and x-y coordinates
            self.fc.axes, self.fc.units = self.sortAxisValues(axisTextImgs)
        except Exception as e:
            print("The error raised is: ", e)
    
    def sortAxisValues(self, axisTextImgs):
        axes = ([0],[0])
        units = ['', '']
        for (img, x, y) in axisTextImgs:
            axis = not (y > 0.8*self.ip.size[0])      # 0: x-axis, 1: y-axis
            text = self.tr.readNumber(img)
            if(len(text) == 0):                         # if no numbers, is unit
                imgs = (img, self.ip.rotate(img, 270))  # take a rotation of the image
                text = self.tr.readUnit(imgs, self.pe.units[axis])
                if(text != None): units[axis] = text   # add to either x-axis or y-axis unit string
            else:                               # numbers are axis values
                val = float(text)               # convert string to float
                if(val != 0.0):                 # filter erroneous values
                    axes[axis].append(val)      # add to either x-axis or y-axis list
        for axis in axes: axis.sort()
        return axes, units

    class graph:
        def __init__(self):
            self.axes = ([0],[0])
            self.units = ['', '']
            self.size = None

    class imageProcessor:
        def __init__(self):
            self.img = None
            self.fileLoc = ""
            self.size = None
            self.imgTypes = ['.png']
        
        def setImage(self, file):
            self.fileLoc = file
            self.setImageDPI(300)   #dpi of 300 is "optimal" for OCR 
            self.img = cv.imread(self.fileLoc)
            self.size = self.img.shape[:2] # (height, width)
            #print("Image Width: " + str(self.size[1]) + " Image Height: " + str(self.size[0]))
        
        def showImage(self, img, label = "Image"):
            cv.imshow(label, img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        def getContours(self, img):
            rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
            dilation = cv.dilate(img, rect_kernel, iterations = 3)
            contours = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
            return contours

        def cropTextFromImage(self):
            gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
            thresh = cv.threshold(gray, 127, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)[1]
            contours = self.getContours(thresh)
            imgs = []
            im2 = self.img.copy()
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                rect = cv.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cropped = im2[y:y + h, x:x + w] # crop the bounding box area
                if(w < 0.5*self.size[1] and h < 0.5*self.size[0]): # check that crop is reasonably small
                    imgs.append((cropped, x, y))
            return imgs

        def rotate(self, img, angle):
            (h, w) = img.shape[:2]
            center = (w/2, h/2)
            M = cv.getRotationMatrix2D(center, angle, scale = 1.0)
            M[0][2] += (h/2) - w/2
            M[1][2] += (w/2) - h/2 # idk what the fuck this does
            # but it works remarkably well
            #note that since the height and width are just flipped, this function only works for 90 deg rots
            rotated = cv.warpAffine(img, M, (h, w)) 
            return rotated

        def setImageDPI(self, dpi):
            im = Image.open(self.fileLoc)
            im.save(self.fileLoc, dpi=(dpi, dpi))

    class textReader:
        def __init__(self, path):
            self.path = path
            pytesseract.tesseract_cmd = self.path

        def readImage(self, img, wl, psm):
            config = "-c tessedit_char_whitelist=" + wl + " --psm " + psm + " --oem 3"
            text = pytesseract.image_to_string(img, config=config)
            text = text.strip('\n')
            return text

        def readNumber(self, img, wl = "0123456789.", psm = "8"):
            return self.readImage(img, wl, psm)
            
        def readUnit(self, imgs, units, psm = "8", align = "horizontal"):
            wl = 'Papm3/hscfin\\"H2O' # tesseract gets confused if there are spaces in this string
            readUnits = (self.readImage(imgs[0], wl, psm), self.readImage(imgs[1], wl, psm))
            trueUnit = None; maxRatio = 0
            for unit in units:
                for readUnit in readUnits:
                    ratio = fuzz.partial_ratio(readUnit, unit)
                    print(readUnit, unit, ratio)
                    if(ratio > maxRatio and ratio > 60):
                        trueUnit = unit
                        maxRatio = ratio
            return trueUnit

    class physicsEngine:
        def __init__(self, Q=1, p_s=1, D=1, P_in=1, L_p=None, w=None):
            self.Q = Q                                      # flow rate [m^3/s]
            self.p_s = p_s                                  # static pressure [Pa]
            self.D = D                                      # diameter [m]
            self.P_in = P_in                                # power consumption [W]
            self.L_p = L_p                                  # sound pressure level [dB (A)]
            self.w = w                                      # rotational velocity [rad/s]
            self.initDerivedParams()
            self.units = [['m3/h', 'm3/s', 'cfm'], ['Pa', 'inH2O', '\"H20']]

        def initDerivedParams(self):
            self.A = self.getA(self.D)                      # area [m^2]
            self.v = self.get_v(self.Q, self.A)             # air velocity [m/s]
            self.p_v = self.get_p_v(self.v)                 # velocity pressure [Pa]
            self.p_t = self.get_p_t(self.p_s, self.p_v)     # total pressure [Pa]
            self.P_out = self.getP_out(self.p_t, self.Q)    # power output [W]
            self.Eff = self.getEff(self.P_in, self.P_out)   # efficiency

        def getA(self, D):
            return math.pi*D**2/4
        
        def get_v(self, Q, A):
            return Q/A
        
        def get_p_v(self, v, rho=1.2):
            # 1.2 kg/m^3 is standard air density
            return 0.5*rho*v**2

        def get_p_t(self, p_s, p_v):
            return p_s + p_v

        def getP_out(self, p_t, Q):
            return p_t*Q

        def getEff(self, P_in, P_out):
            return P_in/P_out
    
    class dataProcessor:
        def __init__(self):
            pass

        def isNumber(self, str):
            try:
                float(str)
                return True
            except:
                return False

    class speech:
        def __init__(self):
            pass

        def ask(self, question):
            answer = ""
            if question == "Fan Curve":
                answer = input("What is the name of the fan curve?")
            return answer

def fcAxesTest(fb):
    fb.acquirefc()
    fb.setAxes()
    print("Flow Rates: " + str(fb.fc.axes[0]) + " " + fb.fc.units[0])
    print("Pressures: " + str(fb.fc.axes[1]) + " " + fb.fc.units[1])

def main():
    fb = fanBot()
    fcAxesTest(fb)

if __name__=="__main__":
    main()