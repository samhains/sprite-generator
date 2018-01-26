import numpy as np
import cv2

MIN_CONTOUR_THRESHOLD = 100


def threshold(image):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    mask = ((r == 128) & (g == 128) & (b == 0))
    image[np.logical_not(mask)] = 255
    image[mask] = 0


def calc_contours(imgray):
    ret,thresh = cv2.threshold(imgray,127,255,0)
    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_THRESHOLD]


def calc_areas(contours):
    return [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > MIN_CONTOUR_THRESHOLD]


def calc_rectangles(contours, areas):
    rectangles = []
    counter = 0
    for i, cnt in enumerate(contours):
        if areas[i] > mean_area*0.8 and areas[i] < mean_area*1.8:
            counter = counter + 1
            if counter:
                rectangles.append(cv2.boundingRect(cnt))


    return rectangles


def draw_rectangles(rectangles, img_url):
    im_raw = cv2.imread(img_url)
    for (x,y,w,h) in rectangles:
        print(x,y,w,h)
        crop_img = im_raw[y:y+h, x:x+w]
        cv2.rectangle(im_raw, (x,y), (x+w,y+h),(0,255,0),2)
        cv2.imshow("cropped", im_raw)
        cv2.waitKey(0)
    print(len(rectangles))
    y = list(set(rectangles))


def read_img(url):
    im = cv2.imread(url)
    threshold(im)
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def adjust_rectangles(rectangles):
    first = rectangles[0]
    # min_y = first[1]
    adjusted_rectangles = []
    first_y = first[1]
    for rect in rectangles:
        y = rect[1]
        # print(y)
        if y > 50 + first_y:
            # print("new row!")
            first_y = y
        else:
            # print("add rect")
            rect = (rect[0], first_y, rect[2], rect[3])

        adjusted_rectangles.append(rect)
    return adjusted_rectangles

def sort_rectangles(rectangles):
    return sorted(rectangles, key=lambda k: [k[1], k[0]])

imgray = read_img('images/beserker_cropped.png')
contours = calc_contours(imgray)
areas = calc_areas(contours)

mean_area = np.mean(areas)
rectangles = calc_rectangles(contours, areas)
rectangles = sort_rectangles(rectangles)
adjusted_rectangles = adjust_rectangles(rectangles)
sorted_rectangles = sort_rectangles(adjusted_rectangles)
draw_rectangles(sorted_rectangles, 'images/beserker_cropped.png')
