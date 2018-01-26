import numpy as np
import cv2
import itertools
im = cv2.imread('images/beserker_cropped.png')
#
# x_dim, y_dim, z_dim = im.shape
# bg_color = im[0,0]
#
# for x, y, c in itertools.product(*map(range, (x_dim, y_dim, z_dim))):
#     rgb = im[x, y]
#     if not np.array_equal(rgb, bg_color):
#         continue
#         # print(rgb)
#     else:
#         im[x,y] =  [0,0,0]
#
# ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
# cv2.imshow(thresh1)
def threshold(image):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # np.where((r < 128) & (g < 128) & (b < 0), 0, 1)
    # mask = (r < 128) & (g < 128) & (b < 0)
    mask = ((r == 128) & (g == 128) & (b == 0))
    # mask =  (b == 0 )
    # new_image = (image == (128, 128, 0)).any(axis=2)
    # np.logical_not(new_image)
    # # print(new_image)
    # # print(new_image.shape)
    # image[new_image] = [255,255,255]
    # new_image = not new_image
    # image[new_image] = [0,0,0]
    #
    # print(image)
    # print(image.shape)
    # mask = (r == 255) & (g == 255) & (b == 255)
    image[np.logical_not(mask)] = 255
    image[mask] = 0


threshold(im)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(im, contours, -1, (0,255,0), 3)
areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100]
contours = [c for c in contours if cv2.contourArea(c) > 100]
rectangles = []
mean_area = np.mean(areas)
# print(contours_reduced[0].shape, contours_reduced[1].shape)
# for contour in contours_reduced:
#     print(contour.shape)
# print(cnt)
cnter = 0

im_raw = cv2.imread('images/beserker_cropped.png')
for i, cnt in enumerate(contours):
    if areas[i] > mean_area*0.8 and areas[i] < mean_area*1.8:
        cnter = cnter + 1
        if cnter:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(im, (x,y), (x+w,y+h),(0,255,0),2)
            crop_img = im_raw[y:y+h, x:x+w]
            # cv2.imshow("cropped", crop_img)

            cv2.imshow("Show",im)
            cv2.waitKey(0)

            rectangles.append((x,y,w,h))
# print('finished', cnter)
# print('rectangles', rectangles)
print(len(rectangles))
# y = list(set(rectangles))
# print('unique rectangles', len(y))

# Show keypoints
cv2.waitKey(0)
