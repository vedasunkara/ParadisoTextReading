from PIL import Image
import numpy as np
import pytesseract
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")
args = vars(ap.parse_args())
confThreshold = 0.05
nmsThreshold = 0.1

def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

def crop(image):
	width, height = np.shape(image)
	width_max = width % 32
	height_max = height % 32

	small_dim = min(width_max, height_max)

	#image = image[0: small_dim * 32, 0: small_dim * 32]
	#print(np.shape(image))
	resized = cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA)

	return resized

def box_image(image):
	net = cv2.dnn.readNet("frozen_east_text_detection.pb")
	image = crop(image)
	width, height = np.shape(image)

	kWinName = "East Model Experiment"
	cv2.namedWindow(kWinName, cv2.WINDOW_NORMAL)

	outNames = np.array([])
	outNames = np.append(outNames, "feature_fusion/Conv_7/Sigmoid")
	outNames = np.append(outNames, "feature_fusion/concat_3")

	blob = cv2.dnn.blobFromImage(image, 1.0, (width, height), (123.68, 116.78, 103.94), True, False)

	net.setInput(blob)
	outs = net.forward(outNames)
	t, _ = net.getPerfProfile()
	label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

	scores = outs[0]
	geometry = outs[1]
	[boxes, confidences] = decode(scores, geometry, confThreshold)

	indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
	for i in indices:
		vertices = cv2.boxPoints(boxes[i[0]])
		for j in range(4):
			vertices[j][0] *= rW
			vertices[j][1] *= rH

		for j in range(4):
			p1 = (vertices[j][0], vertices[j][1])
			p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
			cv.line(image, p1, p2, (0, 255, 0), 1);

	cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

	cv2.imshow(kWinName,image)


def load_image():
    # load the example image and convert it to grayscale
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if args["preprocess"] == "thresh":
	       gray = cv2.threshold(gray, 0, 255,
		         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # make a check to see if median blurring should be done to remove
    # noise
    elif args["preprocess"] == "blur":
	       gray = cv2.medianBlur(gray, 3)

    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    return image, gray, filename


def read_loaded_image(image, gray, filename):
    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    img = Image.open(filename)
    text = pytesseract.image_to_string(img)
    os.remove(filename)
    print(text)

    # show the output images
    cv2.imshow("Image", image)
    cv2.imshow("Output", gray)
    cv2.waitKey(0)

def main():
    image, gray, filename = load_image()

    box_image(gray)

    read_loaded_image(image, gray, filename)

    print("main")

if __name__ == '__main__':
    main()
