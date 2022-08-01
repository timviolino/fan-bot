import os                               # used for interacting with the operating system
import cv2 as cv                        # used for image processing
import math                             # used for calculations                
from pytesseract import pytesseract     # used for reading text from images via OCR
from PIL import Image                   # used for altering image dpi
from fuzzywuzzy import fuzz             # used for making fuzzy comparisons of strings
import numpy as np
# Used for taking screenshots of webpages 
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.options import Options
import requests
import fitz
import io

PATH_TO_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class fanBot:
    def __init__(self):
        self.tr = self.textReader(PATH_TO_TESSERACT)
        self.pe = self.physicsEngine()
        self.ip = self.imageProcessor()
        self.dp = self.dataProcessor()
        self.s = self.speech()
        self.fc = self.graph()
        self.v = self.vision()

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
    
    def setUnitWhitelist(self, units):
        wl = ""
        for axis in units:
            for unit in axis:
                for c in unit:
                    if c not in wl:
                        if c == '"': wl += '\\"'
                        else: wl += c
        return wl

    def sortAxisValues(self, axisTextImgs):
        axes = ([0],[0])
        units = ['', '']
        unitSet = self.pe.units
        unitWL = self.setUnitWhitelist(unitSet)
        for (img, x, y) in axisTextImgs:
            isUnit = False
            axis = not (y > 0.8*self.ip.size[0])        # 0: x-axis, 1: y-axis
            num = self.tr.readNumber(img)
            unit = self.tr.readUnit(img, unitWL)
            if(len(unit) > 1):
                inPhrase = False
                if (len(unit) > 4): inPhrase = True
                isUnit, trueUnit = self.dp.checkIfUnit(unit, unitSet[axis], inPhrase)
                if(isUnit): units[axis] = trueUnit  # add to either x-axis or y-axis unit string
            if(not isUnit and self.dp.isNumber(num)):        # numbers are axis values
                val = float(num)                # convert string to float
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
            rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            dilation = cv.dilate(img, rect_kernel, iterations = 4)
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
                if(w < 0.5*self.size[1] or h < 0.5*self.size[0]): # check that crop is reasonably small
                    if(h > 2*w): cropped = self.rotate(cropped, 270)
                    cropped = cv.resize(cropped, (0,0), fx=3, fy=3)
                    imgs.append((cropped, x, y))
            self.showImage(im2)
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

        def findMatch(self, img, template, method = cv.TM_CCOEFF_NORMED):
            h, w = template.shape[:2]
            res = cv.matchTemplate(img, template, method)
            threshold = 0.05
            loc = np.where(res >= threshold)
            if (len(loc[0]) > 0):
                pt = (loc[1][0], loc[0][0])
                cropped = img[pt[1]:pt[1] + h, pt[0]:pt[0] + w]
                self.showImage(cropped)

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
            
        def readUnit(self, img, unitWL, psm = "8", align = "horizontal"):
            return self.readImage(img, unitWL, psm)
            

    class physicsEngine:
        def __init__(self, Q=1, p_s=1, D=1, P_in=1, L_p=None, w=None):
            self.Q = Q                                      # flow rate [m^3/s]
            self.p_s = p_s                                  # static pressure [Pa]
            self.D = D                                      # diameter [m]
            self.P_in = P_in                                # power consumption [W]
            self.L_p = L_p                                  # sound pressure level [dB (A)]
            self.w = w                                      # rotational velocity [rad/s]
            self.initDerivedParams()
            self.units = [['m3/h', 'm3/s', 'cfm', 'CFM'], ['Pa', 'inH2O', '\"H2O']]

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

        def checkIfUnit(self, text, units, inPhrase, threshold = 70):
            trueUnit = None; maxRatio = 0
            for unit in units:
                ratio = 0
                if(inPhrase): ratio = fuzz.partial_ratio(text, unit)
                else: ratio = fuzz.ratio(text, unit)
                if(ratio > maxRatio and ratio > threshold):
                    trueUnit = unit
                    maxRatio = ratio
            return (maxRatio > threshold), trueUnit

    class speech:
        def __init__(self):
            pass

        def ask(self, question):
            answer = ""
            if question == "Fan Curve":
                answer = input("What is the name of the fan curve?")
            return answer

    class vision:
        def __init__(self):
            pass

        def grabPage(self, pages):
            options = Options()
            options.binary_location = r'C:\Program Files\Mozilla Firefox\firefox.exe'
            driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)
            for page in pages:
                driver.get(page)
                screenshot = driver.save_screenshot('test.png')
            driver.quit()

        def writePDFFromURL(self, url):
            pdf = requests.get(url)
            fileName = "test.pdf"
            open(fileName, "wb").write(pdf.content)
            return fileName

        def parsePDF(self, file):
            pdf = fitz.open(file)
            for i in range(len(pdf)):
                imgs = pdf.get_page_images(i)
                for j in range(len(imgs)):
                    xref = imgs[j][0]
                    baseImg = pdf.extract_image(xref)
                    byteImg = baseImg["image"]
                    ext = baseImg["ext"]
                    image = Image.open(io.BytesIO(byteImg))
                    image.save(open(f"image{i}_{j}.{ext}", "wb"))

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