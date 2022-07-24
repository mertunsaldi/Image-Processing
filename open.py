import cv2 as cv
# read images
im1 = cv.imread('different images/1-1.png')
im2 = cv.imread('different images/1-4.png')
# 3-channel array subtraction of two images
diff = cv.subtract(im1, im2)
# result converted to binary greyscale
diff_grey = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
# found independent differences (contours)
contours, hierarchy = cv.findContours(diff_grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# computed total area of the differences
total_contour_area = sum(cv.contourArea(contour) for contour in contours)
# created an array containing contour area and its weight among the others (namely significance) for each contour
contour_sizes_significance = [(cv.contourArea(contour), contour, cv.contourArea(contour)/total_contour_area*100)
                                       for contour in contours]
# the results will be shown on the second image
result = im2

# !!! INPUT to select differences greater than it (percent)
inp = 5

# selects each contour which has greater significance than input
for size, cont, sig in contour_sizes_significance:
    if sig >= inp:
        # finds the dimensions and placement of the rectangles containing the significant contours and adds to result
        x, y, w, h = cv.boundingRect(cont)
        cv.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

# draws green outline for each difference
cv.drawContours(result, contours, -1, (0, 255, 0), 2)
# shows subtracted greyscale image
cv.imshow('diff', diff_grey)
# shows result
cv.imshow('result', result)
# saves result
cv.imwrite('1-result.png', result)

cv.waitKey(0)

