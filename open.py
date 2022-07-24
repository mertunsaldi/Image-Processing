import cv2 as cv

im1 = cv.imread('different images/1-1.png')
im2 = cv.imread('different images/1-4.png')

diff = cv.subtract(im1, im2)

diff_grey = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

contours, hierarchy = cv.findContours(diff_grey, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

total_contour_area = sum(cv.contourArea(contour) for contour in contours)

contour_sizes_significance = [(cv.contourArea(contour), contour, cv.contourArea(contour)/total_contour_area*100)
                                       for contour in contours]

result = im2

inp = 5

for size, cont, sig in contour_sizes_significance:
    if sig > inp:
        x, y, w, h = cv.boundingRect(cont)
        cv.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv.drawContours(result, contours, -1, (0, 255, 0), 3)
cv.imshow('diff', diff_grey)
cv.imshow('result', result)
cv.imwrite('1-result.png', result)

cv.waitKey(0)

