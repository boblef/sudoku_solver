# import cv2
# import tensorflow as tf
# import numpy as np

# face_cascade = cv2.CascadeClassifier("../xml/haarcascade_frontalface_alt2.xml")
# ds_factor = 0.6


# class Process():
#     def __init__(self, **kwargs):
#         self.frame = kwargs["frame"]
#         self.model = kwargs["model"]

#     def getCellPositions(self, img_PT):
#         # resizing the images to take the shape of the NN model
#         img_PT = cv2.resize(img_PT, (252, 252))
#         # computing position of each cell and storing in an array of arrays
#         cell_positions = []

#         width = img_PT.shape[1]
#         height = img_PT.shape[0]

#         cell_width = width//9
#         cell_height = height//9

#         x1, x2, y1, y2 = 0, 0, 0, 0

#         for i in range(9):
#             y2 = y1 + cell_height
#             x1 = 0
#             for j in range(9):
#                 x2 = x1 + cell_width
#                 current_cell = [x1, x2, y1, y2]
#                 cell_positions.append(current_cell)
#                 x1 = x2
#             y1 = y2
#         return cell_positions

#     def extractSudokuDigits(self, img_PT):
#         # we start looking at the middle of the cell
#         # as this is where the sudoku digit should be at
#         cell_digits, num = [], 0
#         cells = self.getCellPositions(img_PT)
#         for cell in range(len(cells)):
#             num = self.predictDigit(cells[cell], img_PT)
#             cell_digits.append(num)
#         n = 9
#         cell_digits = [cell_digits[i:i+n]
#                        for i in range(0, len(cell_digits), n)]
#         return cell_digits

#     def predictDigit(self, cell, img):
#         pos = []
#         img = cv2.resize(img, (252, 252))
#         img = img[cell[2]+2:cell[3]-3, cell[0]+2:cell[1]-3]
#         contours, hierarchy = cv2.findContours(
#             img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         if len(contours) != 0:

#             for c in contours:
#                 x, y, w, h = cv2.boundingRect(c)
#                 # if the contour is sufficiently large, it must be a digit
#                 # multiplied each number by 9 due to the resized image
#                 if (w < 15 and x > 2) and (h < 25 and y > 2):
#                     # pos = (x,y,x+w,y+h)
#                     pos.append((x, y, x+w, y+h))
#                     break
#         if pos == []:
#             result = 0
#         if pos:
#             img1 = img[(pos[0][1]):(pos[0][3]), (pos[0][0]):(pos[0][2])]
#             # img1 = img[4:25,5:23]
#             # 22-3-2020
#             img1 = cv2.resize(img, (28, 28))
#             img1 = img1.reshape(1, 28, 28, 1)
#             img1 = tf.cast(img1, tf.float32)  # for linux (tf2.0)

#             result = self.prediction(img1)

#         return result

#     def prediction(self, image):
#         """
#         Return a class which the given is being classified.
#         """
#         classes = self.model.predict_classes(image)

#         if classes == [[0]]:
#             return 0
#         elif classes == [[1]]:
#             return 1
#         elif classes == [[2]]:
#             return 2
#         elif classes == [[3]]:
#             return 3
#         elif classes == [[4]]:
#             return 4
#         elif classes == [[5]]:
#             return 5
#         elif classes == [[6]]:
#             return 6
#         elif classes == [[7]]:
#             return 7
#         elif classes == [[8]]:
#             return 8
#         elif classes == [[9]]:
#             return 9

#     # function that takes in points
#     def order_points(self, pts):
#         # initialzie a list of coordinates that will be ordered
#         # such that the first entry in the list is the top-left,
#         # the second entry is the top-right, the third is the
#         # bottom-right, and the fourth is the bottom-left
#         rect = np.zeros((4, 2), dtype="float32")

#         # the top-left point will have the smallest sum, whereas
#         # the bottom-right point will have the largest sum
#         rect[0] = pts[0]
#         rect[2] = pts[2]

#         # now, compute the difference between the points, the
#         # top-right point will have the smallest difference,
#         # whereas the bottom-left will have the largest difference
#         rect[3] = pts[3]
#         rect[1] = pts[1]

#         # return the ordered coordinates
#         return rect

#     def four_point_transform(self, image, pts):
#         # obtain a consistent order of the points and unpack them
#         # individually
#         rect = self.order_points(pts)
#         (tl, tr, br, bl) = rect

#         # compute the width of the new image, which will be the
#         # maximum distance between bottom-right and bottom-left
#         # x-coordiates or the top-right and top-left x-coordinates
#         widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#         widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#         maxWidth = max(int(widthA), int(widthB))

#         # compute the height of the new image, which will be the
#         # maximum distance between the top-right and bottom-right
#         # y-coordinates or the top-left and bottom-left y-coordinates
#         heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#         heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#         maxHeight = max(int(heightA), int(heightB))

#         # now that we have the dimensions of the new image, construct
#         # the set of destination points to obtain a "birds eye view",
#         # (i.e. top-down view) of the image, again specifying points
#         # in the top-left, top-right, bottom-right, and bottom-left
#         # order
#         dst = np.array([
#             [0, 0],
#             [0, maxHeight - 1],
#             [maxWidth - 1, maxHeight - 1],
#             [maxWidth - 1, 0]], dtype="float32")

#         # compute the perspective transform matrix and then apply it
#         M = cv2.getPerspectiveTransform(rect, dst)
#         warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

#         # return the warped image
#         return warped

#     def detectEmptyCell(self, cell, img):
#         pos = []
#         img = cv2.resize(img, (252, 252))
#         img = img[cell[2]+2:cell[3]-3, cell[0]+2:cell[1]-3]
#         contours, _ = cv2.findContours(
#             img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         if len(contours) != 0:

#             for c in contours:
#                 x, y, w, h = cv2.boundingRect(c)
#                 # if the contour is sufficiently large, it must be a digit
#                 # multiplied each number by 9 due to the resized image
#                 if (w < 15 and x > 2) and (h < 25 and y > 2):
#                     # pos = (x,y,x+w,y+h)
#                     pos.append((x, y, x+w, y+h))
#                     break
#         if pos == []:
#             return pos
#         else:
#             return 0


# class ImageProcess(Process):
#     def __init__(self):
#         super().__init__()


# class LiveCamProcess(Process):
#     def __init__(self, **kwargs):
#         super().__init__(kwargs)

#     def placeSudokuDigitsLive(self, img_PT):
#         # we start looking at the middle of the cell as this
#         # is where the sudoku digit should be at
#         # had to reshape the image size to fit the model shape
#         img_PT = cv2.resize(img_PT, (252, 252))
#         # img is got from the grayscale
#         img_color = cv2.resize(self.frame, (252, 252))
#         cells = self.getCellPositions(img_PT)
#         n = 9
#         cr = [cells[i:i+n]
#               for i in range(0, len(cells), n)]  # cr meaning cells reshaped
#         digits = self.extractSudokuDigits(img_PT)
#         # solve(digits)
#         for i in range(len(cr)):
#             for j in range(len(cr[i])):
#                 pos = self.detectEmptyCell(cr[i][j], img_PT)
#                 digit_text = digits[i][j]
#                 if pos == []:
#                     cv2.putText(img_color,
#                                 str(digit_text),
#                                 ((cr[i][j][0]+8),
#                                  (cr[i][j][2]+19)),
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 0.7, (0, 0, 255), 2, cv2.LINE_AA)
#                 else:
#                     continue

#     def main(self):
#         # Connects to your computer's default camera
#         cap = cv2.VideoCapture(0)

#         # Automatically grab width and height from video feed
#         # (returns float which we need to convert to integer for later on!)

#         # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         # to run some lines once
#         flag = True

#         while True:

#             # Capture frame-by-frame
#             ret, frame = cap.read()

#             # Convert the captured frame into grayscale
#             gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

#             # This segment of the code works on the board segment of the frame
#             contours, hierarchy = cv2.findContours(
#                 gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#             # find the biggest area
#             cnt = contours[0]
#             max_area = cv2.contourArea(cnt)

#             for cont in contours:
#                 if cv2.contourArea(cont) > max_area:
#                     cnt = cont
#                     max_area = cv2.contourArea(cont)
#             epsilon = 0.01*cv2.arcLength(cnt, True)
#             poly_approx = cv2.approxPolyDP(cnt, epsilon, True)

#             board_segment = self.four_point_transform(gray, poly_approx)

#             # Applying Gaussian Blurring to the image
#             dst = cv2.GaussianBlur(board_segment, (1, 1), cv2.BORDER_DEFAULT)

#             # Applying Inverse Binary Threshold to the image
#             ret, thresh_inv = cv2.threshold(
#                 dst, 180, 255, cv2.THRESH_BINARY_INV)

#             # Applying Probabilistic Hough Transform on the Binary Image
#             # minLineLength = 100
#             # maxLineGap = 60
#             lines = cv2.HoughLinesP(thresh_inv, 1, np.pi/180, 100,
#                                     minLineLength=100, maxLineGap=10)
#             for a_line in lines:
#                 x1, y1, x2, y2 = a_line[0]
#                 cv2.line(board_segment, (x1, y1), (x2, y2),
#                          (0, 255, 0), 2, cv2.LINE_AA)

#             if flag:
#                 # using neural network model to detect the digits in the image
#                 # new_model = load_model('keras_digit_model.h5')
#                 a = self.extractSudokuDigits(thresh_inv)
#                 print(a)
#                 # solving with backtracking
#                 # solve(a)

#                 # putting back the solved digits on spaces that are empty
#                 # this function won't have the plt.imshow() and also,
#                 # placeSudokuDigitsLive(thresh_inv)
#                 # the colored image would be img

#                 flag = False

#             # overlaying the board segment of the image on the frame
#             x_offset, y_offset = (poly_approx[0][0].tolist()[
#                                   0]), (poly_approx[0][0].tolist()[1])
#             x_end, y_end = \
#                 (x_offset+board_segment.shape[1]),
#             (y_offset+board_segment.shape[0])
#             frame[y_offset:y_end, x_offset:x_end] = board_segment

#             # Display the resulting frame
#             cv2.imshow('frame', frame)

#             # This command let's us quit with the "q" button on a keyboard.
#             # Simply pressing X on the window won't work!
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         # When everything done, release the capture and destroy the windows
#         cap.release()
#         cv2.destroyAllWindows()


# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)

#     def __del__(self):
#         self.video.release()

#     def get_frame(self):
#         # extracting frames
#         ret, frame = self.video.read()
#         frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,
#                            interpolation=cv2.INTER_AREA)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
#         for (x, y, w, h) in face_rects:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             break
#         # encode OpenCV raw frame to jpg and displaying it
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         return jpeg.tobytes()

import cv2
face_cascade = cv2.CascadeClassifier(
    "/Users/koheisuzuki/Desktop/projects/sudoku_solver/xml/haarcascade_frontalface_alt2.xml")
ds_factor = 0.6


class VideoCamera(object):
    def __init__(self):
        """
        Resource: https://github.com/behl1anmol/VideoStreamingFlask
        """
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        image = cv2.resize(image, None, fx=ds_factor,
                           fy=ds_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in face_rects:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
