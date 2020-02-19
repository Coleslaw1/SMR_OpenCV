import cv2 
import matplotlib as plt
import numpy as np
from matplotlib import pyplot as plt

BoltNo = 20
img = cv2.imread(f'Bolts/Bolt{BoltNo}.jpg',1)

def nothing(x):
    pass

def histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.hist(gray.ravel(),256,[0,256]); plt.show()

def create_mask(img):
    #Create a mask filtering al reds and other colours
    blur = cv2.blur(img,(5,5))
    img_hsv=cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower = np.array([50,90,1])
    upper = np.array([200,255,255])
    red_mask = cv2.inRange(img_hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=red_mask)
    histogram(result)
    return result

def color_filter(img_masked):
    #Used to convert to a black and white image which is filter with a binary filter and put through an erosion kernel
    grayImage = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 70, 255, cv2.THRESH_BINARY)
    blackAndWhiteImage = cv2.bitwise_not(blackAndWhiteImage)
    # Define the structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # Apply the opening operation
    opening = cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_OPEN, kernel)
    # Apply the closing operation
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    #blackAndWhiteImage = cv2.dilate(blackAndWhiteImage,kernel,iterations = 3)
    return closing, thresh

def resize(scale_percent, mask):
    #Used to resize the original image (laod of pixels) to a smaller (or bigger) size
    width = int(mask.shape[1] * scale_percent / 100)
    height = int(mask.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA) 
    return resized

def canny_edge(BandW):
    #Detecting the canny_edges which will be used for edge detection
    edged = cv2.Canny(BandW, 30, 200) 
    return edged

def find_contours(edged):
    #Function used to find the contours
    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def extract_data(contours):
    #Used to extract the data from the contours, such as area, angle and size
    cnt = contours[0]
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    area = cv2.contourArea(cnt)
    print (area)
    return cnt, M, cx, cy, area

def draw_edges(cnt, img_form, cx, cy):
    #Drawing the edges on the image so the process becomes visable
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect) 
    box = np.int0(box)
    
    img_contours = cv2.drawContours(img_form,[box],0,(255,255,255),2) 
    img_contours = cv2.drawContours(img_form, contours, -1, (0,255,0), 3)
    cv2.circle(img_contours, (cx, cy), 8, (255, 255, 255), -1)
    return img_contours
    
def show(resized):
    #Function used to present the image
    cv2.imshow('image', resized)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

def run_through_dataset():
    for BoltNo in range (1,22):
        try:
            img = cv2.imread(f'Bolts/Bolt{BoltNo}.jpg',1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plt.hist(gray.ravel(),256,[0,256]); plt.show()
            img_form = resize(20, img)
            Masked = create_mask(img_form)    
            BandW, thresh = color_filter(Masked)
            canny_edges = canny_edge(BandW)
            contours, hierarchy = find_contours(canny_edges)
            cnt, M, cx, cy, area = extract_data(contours)
            #img_contours = draw_edges(cnt, img_form)  
            print (BoltNo)
        except:
            pass

img_form = resize(20, img)
Masked = create_mask(img_form)    
BandW, thresh = color_filter(Masked)
canny_edges = canny_edge(BandW)
contours, hierarchy = find_contours(canny_edges)
cnt, M, cx, cy, area = extract_data(contours)
img_contours = draw_edges(cnt, img_form, cx, cy)

#show(img_contours)



