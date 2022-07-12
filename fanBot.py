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

    def acquireFanCurve(self):
        file = self.s.ask("Fan Curve")
        self.ip.setImage(file)

    def getAxes(self):
        flowRates = [0]
        pressures = [0]
        try:
            textImgs = self.ip.cropTextFromImage() # returns cropped image with text and x-y coordinates
            for (img, x, y) in textImgs:
                text = self.tr.readImage(img)
                print(text, x, y)
                if(not self.dp.isNumber(text)): continue
                val = float(text)
                if(val == 0.0): continue
                if(y > 0.8*self.ip.h):
                    flowRates.append(val)
                else:
                    pressures.append(val)
        except Exception as e:
            print("The error raised is: ", e)
        return sorted(flowRates), sorted(pressures)

    class imageProcessor:
        def __init__(self):
            self.img = None
            self.fileLoc = ""
            self.w = None
            self.h = None
        
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
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                rect = cv.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cropped = im2[y:y + h, x:x + w] # crop the bounding box area
                if(w < 0.5*self.w and h < 0.5*self.h): # check that crop is reasonably small
                    imgs.append((cropped, x, y))
            return imgs

        def cropTextInRangeThresholding(self):
            hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, np.array([0, 0, 244]), np.array([179, 35, 255]))
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 3))
            dilated = cv.dilate(mask, kernel, iterations=1)
            thresh = cv.bitwise_and(dilated, mask)
            self.showImage(thresh)

        def setImageDPI(self, dpi):
            im = Image.open(self.fileLoc)
            im.save(self.fileLoc, dpi=(dpi, dpi))

    class textReader:
        def __init__(self, path):
            self.path = path
            pytesseract.tesseract_cmd = self.path
        
        def readImage(self, img):
            whitelist = "-c tessedit_char_whitelist=0123456789. --psm 10 --oem 3"
            text = pytesseract.image_to_string(img, config=whitelist)
            text = text.strip('\n')
            return text

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