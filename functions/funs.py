import cv2
import numpy as np

def gstreamer_pipeline(
    #capture_width=1280,
    #capture_height=720,
    capture_width=480,
    capture_height=480,
    #display_width=1280,
    #display_height=720,
    display_width=480,
    display_height=480,
    framerate=60,
    flip_method=6,):

    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

#######################################################################################################

def img2double(img_link):
    # convert img to matris (between 0 and 1) for subtraction correctly
    # TODO: must update function in 'stackoverflow.com/questions/29100722/equivalent-im2double-function-in-opencv-python'

    im = cv2.imread(img_link)
    # min_val = np.min (im.ravel())
    # max_val = np.max (im.ravel())
    # out = (im.astype('float') - min_val) / (max_val-min_val)
    out = cv2.normalize(im.astype('float'), None, 0.0,1.0,cv2.NORM_MINMAX)
    print out
    # return out

#######################################################################################################

def start_shoting_from_object():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline())
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        shot_counter = 1
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)
            
            if(shot_counter%100 == 1):
                cv2.imwrite('images/items/ax.png', img)
                
            shot_counter = shot_counter +1 
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

#######################################################################################################

def shot_from_mold():
    print(gstreamer_pipeline())
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        shot_counter = 1
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)
            if(shot_counter < 50):
                if(shot_counter == 40):
                    # be dalil inke axhaye avlale shot noore kafi nadarand shote 20 ra melak gharar dadim
                    cv2.imwrite('images/mold/mold.png', img)
                    break
                shot_counter = shot_counter +1 
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

#######################################################################################################

def show_mold():
    from PIL import Image
    img = Image.open('images/mold/mold.png')
    img.show()

#######################################################################################################

def live_camera_without_capture():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline())
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

#######################################################################################################

def start_subtraction():

    refrence_img_link = 'images/mold/mold.png'
    new_img_link = 'images/items/ax.png'

    # get images size
    read_refrence_img = cv2.imread(refrence_img_link)
    height_refrence_img = np.size(read_refrence_img, 0)
    width_refrence_img = np.size(read_refrence_img, 1)

    read_new_img = cv2.imread(new_img_link)
    height_new_img = np.size(read_new_img, 0)
    width_new_img = np.size(read_new_img, 1)

    if height_refrence_img != height_new_img:
	    print ('error: height_refrence_img != height_new_img')
    if width_refrence_img != width_new_img:
	    print ('error: width_refrence_img != width_new_img')

    # init images
    new_image = cv2.imread(new_img_link)
    refrence = cv2.imread(refrence_img_link)
    

    # subtract the images
    subtracted = cv2.subtract( new_image, refrence)

    # TO show the output
    cv2.imshow('image', subtracted)

    # save result
    cv2.imwrite('result.jpg', subtracted)

    # To close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#######################################################################################################

def otsu_thresholding():
    # convert img to binary for show high contrast
    image1 = cv2.imread('result.jpg')
  
    # cv2.cvtColor is applied over the
    # image input with applied parameters
    # to convert the image in grayscale
    img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  
    # applying Otsu thresholding
    # as an extra flag in binary 
    # thresholding     
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + 
                                            cv2.THRESH_OTSU)     
  
    # the window showing output image         
    # with the corresponding thresholding         
    # techniques applied to the input image    
    cv2.imshow('Otsu Threshold', thresh1)         
       
    # De-allocate any associated memory usage         
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

#######################################################################################################