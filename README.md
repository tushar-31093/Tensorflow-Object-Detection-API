
# Tensorflow Object Detection API
Creating accurate machine learning models capable of localizing and identifying
multiple objects in a single image remains a core challenge in computer vision.
The TensorFlow Object Detection API is an open source framework built on top of
TensorFlow that makes it easy to construct, train and deploy object detection
models.  

# Pre-requisites:
Download the Tensorflow Object Detection API from <a href="https://github.com/tensorflow/models/tree/master/research/object_detection"> here </a>.
After you have download it please follow the instructions carefully and modify the file accordingly.

# Steps are as follows:
To begin, we're going to modify the notebook first by converting it to a .py file. If you want to keep it in a notebook, that's fine too. 
To convert, you can go to file > save as > python file. Once that's done, you're going to want to comment out the get_ipython().magic('matplotlib inline') line.

Next, we're going to bring in the Python Open CV wrapper:

If you do not have OpenCV installed, you will need to grab it. See the OpenCV introduction for instructions.

import cv2

cap = cv2.VideoCapture(0)
This will prepare the cap variable to access your webcam.

Next, you're going to replace the following code:

    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
With:

    while True:
      ret, image_np = cap.read()
Finally, replace the following:

      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      plt.show()
With:

      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

		
That's it! 