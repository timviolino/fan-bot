import os
import cv2 as cv
import math
from pytesseract import pytesseract
import numpy as np
from PIL import Image

GRAPH_FILE = "graph300.png"
PATH_TO_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class fanBot:
    def __init__(self):
        self.tr = self.textReader(PATH_TO_TESSERACT)
        self.pe = None
        self.ip = self.imageProcessor()
        self.dp = self.dataProcessor()
        self.s = self.speech()
        self.fanCurve = self.graph()

    def acquireFanCurve(self):
        fileType = ''
        while(fileType not in self.ip.imgTypes):
            file = self.s.ask("Fan Curve")
            fileType = os.path.splitext(file)[1]
        self.ip.setImage(file)

    def getAxes(self):
        axes = self.fanCurve.getAxes()
        self.fanCurve.setSize(self.ip.w, self.ip.h)
        try:
            textImgs = self.ip.cropTextFromImage() # returns cropped image with text and x-y coordinates
            for (img, x, y, w, h) in textImgs:
                text = self.tr.readNumber(img)
                if(len(text) == 0): text = self.tr.readWord(img)
                print(text)
                isNumber = self.dp.isNumber(text)
                axis = not (y > 0.8*self.ip.h)
                if(isNumber): # numbers are axis values
                    val = float(text)
                    if(val == 0.0): continue
                    axes[axis].append(val)
                else: # words are units
                    self.fanCurve.units[axis] = text
        except Exception as e:
            print("The error raised is: ", e)
        for axis in axes: axis.sort()
        return axes
    
    class graph:
        def __init__(self):
            self.axes = ([0],[0])
            self.units = ['', '']
            self.pixels = [0, 0]
        
        def getAxes(self):
            return (self.axes[0], self.axes[1])

        def setSize(self, w, h):
            self.pixels = [w, h]

    class imageProcessor:
        def __init__(self):
            self.img = None
            self.fileLoc = ""
            self.w = None
            self.h = None
            self.imgTypes = ['.png']
        
        def setImage(self, file):
            self.fileLoc = file
            self.setImageDPI(300)   #dpi of 300 is "optimal" for OCR 
            self.img = cv.imread(self.fileLoc)
            self.h, self.w = self.img.shape[:2]
            print("Image Width: " + str(self.w) + " Image Height: " + str(self.h))
        
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
            count = 0
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                rect = cv.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cropped = im2[y:y + h, x:x + w] # crop the bounding box area
                if(w < 0.5*self.w and h < 0.5*self.h): # check that crop is reasonably small
                    if(w < h): cropped = self.rotate(cropped)
                    imgs.append((cropped, x, y, w, h))
                    count += 1
                    self.showImage(cropped)
            print(count)
            return imgs

        def rotate(self, img):
            (h, w) = img.shape[:2]
            center = (w/2, h/2)
            M = cv.getRotationMatrix2D(center, angle = 270, scale = 1.0)
            rotated = cv.warpAffine(img, M, (w, h))
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
            
        def readWord(self, img, wl = "qwertyuiopasdfghjklzxcvbnm", psm = "8", align = "horizontal"):
            if (align == "vertical"): psm = "5"
            return self.readImage(img, wl, psm)

    class physicsEngine:
        def __init__(self, Q, p_s, D, P_in, L_p=None, w=None):
            self.Q = Q                                      # flow rate [m^3/s]
            self.p_s = p_s                                  # static pressure [Pa]
            self.D = D                                      # diameter [m]
            self.P_in = P_in                                # power consumption [W]
            self.L_p = L_p                                  # sound pressure level [dB (A)]
            self.w = w                                      # rotational velocity [rad/s]
            self.A = self.getA(self.D)                      # area [m^2]
            self.v = self.get_v(self.Q, self.A)             # air velocity [m/s]
            self.p_v = self.get_p_v(self.v)                 # velocity pressure [Pa]
            self.p_t = self.get_p_t(self.p_s, self.p_v)     # total pressure [Pa]
            self.P_out = self.getP_out(self.p_t, self.Q)    # power output [W]
            self.Eff = self.getEff(self.P_in, self.P_out)   # efficiency

        def getA(D):
            return math.pi*D**2/4
        
        def get_v(Q, A):
            return Q/A
        
        def get_p_v(v, rho=1.2):
            # 1.2 kg/m^3 is standard air density
            return 0.5*rho*v**2

        def get_p_t(p_s, p_v):
            return p_s + p_v

        def getP_out(p_t, Q):
            return p_t*Q

        def getEff(P_in, P_out):
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

def main():
    fb = fanBot()
    fb.acquireFanCurve()
    (flowRates, pressures) = fb.getAxes()
    print("Flow Rates: " + str(flowRates))
    print("Pressures: " + str(pressures))

if __name__=="__main__":
    main()