# import the necessary packages
import tensorflow
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from smbus2 import SMBus
from mlx90614 import MLX90614
import RPi.GPIO as GPIO
from time import sleep
#import notify2
#import subprocess



#initialize temperature sensor bus and gpio
bus = SMBus(1)
sensor = MLX90614(bus, address=0x5a)

#LED setup
greenLed = 8
redLed = 7
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(greenLed, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(redLed, GPIO.OUT, initial=GPIO.HIGH)





#Buzzer setup
buzz = 23
GPIO.setup(buzz, GPIO.OUT)
GPIO.output(buzz, GPIO.LOW)

#IR sensor setup
ir = 10
GPIO.setup(ir, GPIO.IN)

# Define some device parameters
I2C_ADDR  = 0x27 # I2C device address
LCD_WIDTH = 16   # Maximum characters per line

# Define some device constants
LCD_CHR = 1 # Mode - Sending data
LCD_CMD = 0 # Mode - Sending command

LCD_LINE_1 = 0x80 # LCD RAM address for the 1st line
LCD_LINE_2 = 0xC0 # LCD RAM address for the 2nd line
LCD_LINE_3 = 0x94 # LCD RAM address for the 3rd line
LCD_LINE_4 = 0xD4 # LCD RAM address for the 4th line

LCD_BACKLIGHT  = 0x08  # On
#LCD_BACKLIGHT = 0x00  # Off

ENABLE = 0b00000100 # Enable bit

# Timing constants
E_PULSE = 0.0005
E_DELAY = 0.0005

def lcd_init():
  # Initialise display
  lcd_byte(0x33,LCD_CMD) # 110011 Initialise
  lcd_byte(0x32,LCD_CMD) # 110010 Initialise
  lcd_byte(0x06,LCD_CMD) # 000110 Cursor move direction
  lcd_byte(0x0C,LCD_CMD) # 001100 Display On,Cursor Off, Blink Off 
  lcd_byte(0x28,LCD_CMD) # 101000 Data length, number of lines, font size
  lcd_byte(0x01,LCD_CMD) # 000001 Clear display
  time.sleep(E_DELAY)


def lcd_byte(bits, mode):
  # Send byte to data pins
  # bits = the data
  # mode = 1 for data
  #        0 for command

  bits_high = mode | (bits & 0xF0) | LCD_BACKLIGHT
  bits_low = mode | ((bits<<4) & 0xF0) | LCD_BACKLIGHT

  # High bits
  bus.write_byte(I2C_ADDR, bits_high)
  lcd_toggle_enable(bits_high)

  # Low bits
  bus.write_byte(I2C_ADDR, bits_low)
  lcd_toggle_enable(bits_low)
  
def lcd_toggle_enable(bits):
  # Toggle enable
  time.sleep(E_DELAY)
  bus.write_byte(I2C_ADDR, (bits | ENABLE))
  time.sleep(E_PULSE)
  bus.write_byte(I2C_ADDR,(bits & ~ENABLE))
  time.sleep(E_DELAY)

  
def lcd_string(message,line):
  # Send string to display

  message = message.ljust(LCD_WIDTH," ")

  lcd_byte(line, LCD_CMD)

  for i in range(LCD_WIDTH):
    lcd_byte(ord(message[i]),LCD_CHR)
"""
def sendMessage(title, msg):
    subprocess.Popen(['notify-send', msg])
    return"""


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = tensorflow.keras.preprocessing.image.img_to_array(face)
			face = tensorflow.keras.applications.mobilenet_v2.preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
		
		

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

   
    

#Apply Algorithm
def applyLogic(label):
    #pwm.start(0)
    temp = getTempData()
    if temp >= 30:
        #print("Greater")
        #GPIO.output(buzz, GPIO.HIGH)
        GPIO.output(greenLed, GPIO.HIGH)
        lcd_string("Temperature:"+str(temp),LCD_LINE_1)
        
        sleep(1)
    elif (label=="No Mask"):
        GPIO.output(redLed, GPIO.HIGH)
        GPIO.output(greenLed, GPIO.LOW)
        GPIO.output(buzz, GPIO.HIGH)
        lcd_string("No Mask",LCD_LINE_1)
    else:
        GPIO.output(buzz, GPIO.LOW)
        GPIO.output(greenLed, GPIO.HIGH)
        GPIO.output(redLed, GPIO.LOW)
        lcd_string("Mask On",LCD_LINE_1)
        lcd_string("PROCEED",LCD_LINE_2)
        

def getTempData():
    temp = sensor.get_obj_temp()
    return temp
"""
def closeEverything():
    GPIO.output(redLed, GPIO.LOW)
    GPIO.output(greenLed, GPIO.LOW)
    GPIO.output(buzz, GPIO.LOW)
    closeGate()"""

def detect_mask(locs, preds, frame):
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
            
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        #print(label)

        # include the probability in the label
        #label_out = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        #temperature sensor data
        temp = getTempData()
        #temp = sensor.get_object_1()
        person_temp = "Temp: {:.1f}".format(temp)
        
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.putText(frame, person_temp, (endX-10, endY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        #dist = GPIO.input(ir)

        #if dist == 0:
        applyLogic(label)
        #else:
            #closeEverything()
        



# loop over the frames from the video stream
def run_video(detect_and_predict_mask):
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        #cv2.normalize(frame, frame,0,255, cv2.NORM_MINMAX)

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            

        # loop over the detected face locations and their corresponding
        # locations
        detect_mask(locs, preds, frame)
        
        # output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        #Breaking the loop
        if key == ord("q"):
            break


    cv2.destroyAllWindows()
    #pwm.stop()
    GPIO.cleanup()
    vs.stop()
    


#main function
if __name__=="__main__":
    lcd_init()
    lcd_string("Welcome",LCD_LINE_1)
    sleep(1)
    # load our serialized face detector model from disk
    prototxtPath = r"/home/project/deploy.prototxt"
    weightsPath = r"/home/project/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = tensorflow.keras.models.load_model("/home/project/mask_model11.model")
    
    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    
    run_video(detect_and_predict_mask)
