import cv2
import numpy as np


class VideoCamera():
    def __init__(self, model):
        """
        Resource: https://github.com/behl1anmol/VideoStreamingFlask
        """
        self.model = model
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def predict(self, image):
        """
        Return a class which the given is being classified.
        """
        classes = self.model.predict_classes(image)

        if classes == [0]:
            return 0
        elif classes == [1]:
            return 1
        elif classes == [2]:
            return 2
        elif classes == [3]:
            return 3
        elif classes == [4]:
            return 4
        elif classes == [5]:
            return 5
        elif classes == [6]:
            return 6
        elif classes == [7]:
            return 7
        elif classes == [8]:
            return 8
        elif classes == [9]:
            return 9

    def sudoku_cv(self):
        #
        # 1. preprocessing
        #
        success, frame = self.video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            src=gray, maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
        blurred = cv2.medianBlur(binary, ksize=3)

        #
        # 2. try to find the sudoku
        #
        contours, _ = cv2.findContours(image=cv2.bitwise_not(blurred),
                                       mode=cv2.RETR_LIST,
                                       method=cv2.CHAIN_APPROX_SIMPLE)
        sudoku_area = 0
        sudoku_contour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if (0.7 < float(w) / h < 1.3     # aspect ratio
                    and area > 150 * 150     # minimal area
                    and area > sudoku_area   # biggest area on screen
                    and area > .5 * w * h):  # fills bounding rect
                sudoku_area = area
                sudoku_contour = cnt

        #
        # 3. separate sudoku from background
        #
        if sudoku_contour is not None:

            # approximate the contour with connected lines
            perimeter = cv2.arcLength(curve=sudoku_contour, closed=True)
            approx = cv2.approxPolyDP(curve=sudoku_contour,
                                      epsilon=0.1 * perimeter,
                                      closed=True)

            if len(approx) == 4:
                # successfully approximated
                # we now transform the sudoku to a fixed size 450x450
                # plus 50 pixel border and remove the background

                # create empty mask image
                mask = np.zeros(gray.shape, np.uint8)
                # fill a the sudoku-contour with white
                cv2.drawContours(mask, [sudoku_contour], 0, 255, -1)
                # invert the mask
                mask_inv = cv2.bitwise_not(mask)
                # the blurred picture is already thresholded so this step shows
                # only the black areas in the sudoku
                separated = cv2.bitwise_or(mask_inv, blurred)

                # create a perspective transformation matrix.
                # "square" defines the target dimensions (450x450).
                # The image we warp "separated" in
                # has bigger dimensions than that (550x550) to assure that no
                # pixels are cut off accidentially on twisted images
                square = np.float32(
                    [[50, 50], [500, 50], [50, 500], [500, 500]])
                # api needs conversion
                approx = np.float32([i[0] for i in approx])
                # sort the approx points to match the points defined in square
                approx = self.sort_grid_points(approx)

                m = cv2.getPerspectiveTransform(approx, square)
                transformed = cv2.warpPerspective(separated, m, (550, 550))

                #
                # 4. get crossing points to determine grid buckling
                #

                # 4.1 vertical lines
                #

                # sobel x-axis
                sobel_x = cv2.Sobel(transformed, ddepth=-1, dx=1, dy=0)

                # closing x-axis
                # vertical kernel
                kernel_x = np.array([[1]] * 20, dtype='uint8')
                dilated_x = cv2.dilate(sobel_x, kernel_x)
                closed_x = cv2.erode(dilated_x, kernel_x)
                _, threshed_x = cv2.threshold(closed_x, thresh=250, maxval=255,
                                              type=cv2.THRESH_BINARY)

                # generate mask for x
                contours, _ = cv2.findContours(image=threshed_x,
                                               mode=cv2.RETR_LIST,
                                               method=cv2.CHAIN_APPROX_SIMPLE)
                from functools import cmp_to_key
                # sort contours by height
                sorted_contours = sorted(
                    contours, key=cmp_to_key(self.cmp_height))

                # fill biggest 10 contours on mask (white)
                mask_x = np.zeros(transformed.shape, np.uint8)
                cv2.drawContours(mask_x, sorted_contours[:10], -1, 255, -1)

                # 4.2 horizontal lines
                #

                # this is essentially the same procedure as for the x-axis
                # sobel y-axis
                sobel_y = cv2.Sobel(transformed, ddepth=-1, dx=0, dy=1)

                # closing y-axis
                # horizontal krnl
                kernel_y = np.array([[[1]] * 20], dtype='uint8')
                dilated_y = cv2.dilate(sobel_y, kernel_y)
                closed_y = cv2.erode(dilated_y, kernel_y)
                _, threshed_y = cv2.threshold(closed_y, 250, 255,
                                              cv2.THRESH_BINARY)

                # generate mask for y
                contours, _ = cv2.findContours(image=threshed_y,
                                               mode=cv2.RETR_LIST,
                                               method=cv2.CHAIN_APPROX_SIMPLE)
                sorted_contours = sorted(
                    contours, key=cmp_to_key(self.cmp_width))

                # fill biggest 10 on mask
                mask_y = np.zeros(transformed.shape, np.uint8)
                cv2.drawContours(mask_y, sorted_contours[:10], -1, 255, -1)

                #
                # 4.3 close the grid
                #
                dilated_ver = cv2.dilate(mask_x, kernel_x)
                dilated_hor = cv2.dilate(mask_y, kernel_y)
                # now we have the single crossing points
                # as well as the complete grid
                grid = cv2.bitwise_or(dilated_hor, dilated_ver)
                crossing = cv2.bitwise_and(dilated_hor, dilated_ver)
                #
                # 5. sort crossing points
                #
                contours, _ = cv2.findContours(image=crossing,
                                               mode=cv2.RETR_LIST,
                                               method=cv2.CHAIN_APPROX_SIMPLE)
                # a complete sudoku must have exactly 100 crossing points
                if len(contours) == 100:
                    # take the center points of the bounding rects
                    # of the crossing points.
                    # This should be precise enough, calculating the
                    # moments is not necessary.
                    crossing_points = np.empty(shape=(100, 2))
                    for n, cnt in enumerate(contours):
                        x, y, w, h = cv2.boundingRect(cnt)
                        cx, cy = (x + .5 * w, y + .5 * h)
                        crossing_points[n] = [int(cx), int(cy)]
                    sorted_cross_points = self.sort_grid_points(
                        crossing_points)
                    # show the numbers next to the points
                    for n, p in enumerate(sorted_cross_points):
                        x, y = int(p[0][0]), int(p[0][1])
                        grid = self.draw_str(grid, x, y, str(n))

                    #
                    # 6. Solve the sudoku
                    #
                    sudoku_grid = self.solve_sudoku_ocr(
                        transformed, sorted_cross_points)
                    ret, jpeg = cv2.imencode('.jpg', grid)
                    return jpeg.tobytes(), sudoku_grid
                else:
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    return jpeg.tobytes(), None
            else:
                # cv2.drawContours(frame, [sudoku_contour], 0, 255)
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes(), None
        else:
            # cv2.drawContours(frame, [sudoku_contour], 0, 255)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes(), None

    def sort_grid_points(self, points):
        """
        Given a flat list of points (x, y), this function returns the list of
        points sorted from top to bottom, then groupwise from left to right.
        We assume that the points are nearly equidistant and have the form of a
        square.
        """
        w, _ = points.shape
        sqrt_w = int(np.sqrt(w))
        # sort by y
        points = points[np.argsort(points[:, 1])]
        # put the points in groups (rows)
        points = np.reshape(points, (sqrt_w, sqrt_w, 2))
        # sort rows by x
        points = np.vstack([row[np.argsort(row[:, 0])] for row in points])
        # undo shape transformation
        points = np.reshape(points, (w, 1, 2))
        return points

    def cmp_height(self, x, y):
        """used for sorting by height"""
        _, _, _, hx = cv2.boundingRect(x)
        _, _, _, hy = cv2.boundingRect(y)
        return hy - hx

    def cmp_width(self, x, y):
        """used for sorting by width"""
        _, _, wx, _ = cv2.boundingRect(x)
        _, _, wy, _ = cv2.boundingRect(y)
        return wy - wx

    def draw_str(self, dst, x, y, s):
        """
        Draw a string with a dark contour
        """
        cv2.putText(dst, s, (x + 1, y + 1),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0),
                    thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(dst, s, (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255),
                    lineType=cv2.LINE_AA)
        return dst

    def solve_sudoku_ocr(self, src, crossing_points):
        """
        Split the rectified sudoku image into smaller pictures of letters only.
        Then perform ocr on the letter images,
        create and solve the sudoku using the Sudoku class.
        """
        numbers = []
        # enumerate all the crossing points except the ones
        # on the far right border to get the single cells
        for i, pos in enumerate(
                [pos for pos in range(89) if (pos + 1) % 10 != 0]):

            # warp the perspective of the cell to match a square.
            # the target image "transformed" is slightly smaller
            # than "square" to cut off noise on the borders
            square = np.float32([[-10, -10], [40, -10], [-10, 40], [40, 40]])
            # get the corner points for the cell i
            quad = np.float32([crossing_points[pos],
                               crossing_points[pos + 1],
                               crossing_points[pos + 10],
                               crossing_points[pos + 11]])

            matrix = cv2.getPerspectiveTransform(quad, square)
            transformed = cv2.warpPerspective(src, matrix, (28, 28))

            transformed = cv2.resize(transformed, (28, 28))
            transformed = transformed.reshape((1, 28, 28, 1))
            transformed = transformed.astype('float32')

            # Get sum of all the pixels in the cell
            # If sum value is large it means the cell is blank
            pixel_sum = np.sum(transformed)

            # 190000.0 is the threshold. Bigger than this is an empty cell
            # Threshold can be modified
            if pixel_sum > 190000.0:
                ocr_text = 0
            else:
                # Predict the digit in the cell
                ocr_text = self.predict(transformed)

            numbers.append(int(ocr_text))

        # draw the recognized numbers into the image
        for x in range(9):
            for y in range(9):
                number = numbers[y * 9 + x]
                if not number == 0:
                    self.draw_str(src, 75 + x * 50, 75 + y * 50, str(number))

        # try to solve the sudoku using the Sudoku class
        try:
            reshaped = np.array(numbers).reshape(9, 9)
            print(reshaped)

            ######
            # Solve sudoku
            ######
        except:
            # no solutions found
            pass
        return reshaped
