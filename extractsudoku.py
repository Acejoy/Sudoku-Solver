import os
import cv2
import operator
import numpy as np
from PIL import Image as im 
import matplotlib.pyplot as plt

def resizeImg(pic ,scale) :
    w, h = pic.shape[1], pic.shape[0]
    dsize = (w*scale//100, h*scale//100)
    pic = cv2.resize(pic, dsize)

    return pic

def displayImage(dictImages):
    
    for mes, im in dictImages.items():
        cv2.imshow(mes, im)
        # cv2.imwrite('./Img/'+mes+'.jpg', im)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None
def multiImages(listImages):
    fig = plt.figure()

    for idx, img in enumerate(listImages):
        plt.subplot(9,9, idx+1)
        plt.imshow(img, cmap='gray')
        #plt.title(idx+1)
    plt.show()

def getSquareContours(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.GaussianBlur(img,(5,5),0)
    th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
    dilated = cv2.dilate(th, (13,13), iterations=1)
    edges = cv2.Canny(dilated, 100, 200)
    contours, hierarchy = cv2.findContours(edges,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)   
    #print(cntsSorted[0].shape)
    return cntsSorted

def getCornersSquare(reqContour) :
    # Bottom-right point has the largest (x + y) value.
    # Top-left has point smallest (x + y) value.
    # The bottom-left point has the smallest (x — y) value.
    # The top-right point has the largest (x — y) value.
    reqContour = reqContour.reshape(-1,2)
    #print(reqContour.shape)
    bottom_right, _ = max(enumerate([pt[0]+pt[1] for pt in reqContour]), key = operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0]-pt[1] for pt in reqContour]), key = operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0]-pt[1] for pt in reqContour]), key = operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0]+pt[1] for pt in reqContour]), key = operator.itemgetter(1))

    cornerPoints = [reqContour[top_left], reqContour[top_right], reqContour[bottom_right], reqContour[bottom_left]]
    return cornerPoints

def addPoints(img, points) :

    for p in points:
        img = cv2.circle(img, (p[0], p[1]), 3, (255,0,0), thickness=-3)
    return img

def getL2Distance(point1, point2) :
    diffX = point1[0] -point2[0]
    diffY = point1[1] -point2[1]

    return np.sqrt((diffX**2 + diffY**2))

def getCroppedImage(img, points):
    
    top_left = points[0]
    top_right = points[1]
    bottom_right = points[2]
    bottom_left = points[3]
    src = np.array([top_left, top_right,
                 bottom_right, bottom_left], dtype='float32')
    
    side = max([  getL2Distance(bottom_right, top_right), 
            getL2Distance(top_left, bottom_left),
            getL2Distance(bottom_right, bottom_left),   
            getL2Distance(top_left, top_right) ])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1],
                                 [0, side - 1]], dtype='float32')

    m = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, m, (int(side), int(side)))

    return img
def extractDigits(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3),np.uint8)
    thresholded = th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
    processedImage = cv2.erode(thresholded, kernel, iterations=1)
    
    digits=[]
    #print(thresholded.shape)
    displayImage({'inverted':processedImage})
    side = processedImage.shape[0]//9
    # After getting the cropped digit , we chip away its side as
    # the digit lies in center and the sides may contain borders of the sudoko puzzle
    chipValue = int(0.15*side)

    for i in range(9):
        for j in range(9):
            p1 = (i*side, j*side)
            p2 = ((i+1)*side, (j+1)*side)
            croppedDigit = processedImage[p1[0]:p2[0], p1[1]:p2[1]]
            # we black all the borders to take care of borders from sudoku puzzle
            croppedDigit[0:chipValue, 0:side] = 0
            croppedDigit[0:side, 0:chipValue] = 0
            croppedDigit[side-chipValue:side, 0:side] = 0
            croppedDigit[0:side, side-chipValue:side] = 0
            digits.append(croppedDigit)


    return digits

def saveExtractedDigits(listDigits):
    #save some particular digits from the list
    directory = './digits'
    #saveIndices = [1,2,3,6,9,22,26,57,33,76, 4, 7, 10, 75, 49, 46, 47]
    for idx, digitImg in enumerate(listDigits):
        #if idx in saveIndices:
        filename = os.path.join(directory, str(idx)+'.jpg')
        data = im.fromarray(digitImg)
        data.save(filename)

    return None

orig_image = cv2.imread('./pictures/ex2.jpg')
orig_image = resizeImg(orig_image,200)

sudokuContours = getSquareContours(orig_image)
puzzleContour = cv2.drawContours(orig_image.copy(), sudokuContours,0, (0,255,0), 2)


p1 = getCornersSquare(sudokuContours[0])
puzzleContour = addPoints(puzzleContour, p1)
#print(p1)
#print("-->",sudokuContours[0].shape)
solvingArea = getCroppedImage(orig_image.copy(), p1)
d = {'original image':orig_image,'contoured image': puzzleContour, 'cropped image':solvingArea }
#print(solvingArea.shape)
displayImage(d)
digitsExtracted = extractDigits(solvingArea.copy())
multiImages(digitsExtracted)
saveExtractedDigits(digitsExtracted)