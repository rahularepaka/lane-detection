
# Real-time Lane Detection

We will train and construct a model that helps detect lanes on straight and curved roads based on the raw pixels captured by the single front-facing camera.

<img width="716" alt="Screenshot 2022-05-31 025842" src="https://user-images.githubusercontent.com/10435564/171062604-bde6d4ec-a364-4805-9e39-e3ce9df48f20.png">


## Requirements

To install all the dependencies in this project run

```bash
  pip install -r requirements.txt
```

## Deploymnt

To run this project, you will need to add the following command

```bash
git clone https://github.com/rahularepaka/lane-detection.git
```

```bash
cd lane-detection
```

```bash
python main.python
```

## Libraries 

- OpenCV
- Numpy
- Tensorflow
- Keras

## Dataset

This project demonstrates lane detection using a single image from a road dataset. The lanes are marked by a solid white line (on the right) and alternating short line segments with dots (on the left).

![solidWhiteCurve](https://user-images.githubusercontent.com/10435564/171062709-48773593-879f-429d-85c1-eebfcefb1ca8.jpg)

## Computer Vision Techniques

**Canny Edge Detection**
```bash
  def canny(img):
    if img is None:
        cap.release()
        cv.destroyAllWindows()
        exit()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv.Canny(gray, 50, 150)
    return canny
```
<img width="370" alt="canny" src="https://user-images.githubusercontent.com/10435564/171062798-d1c35e9f-758f-4220-8c60-dac78a0b9ffe.png">


```bash
def edge_detection(image):
    edges = cv.Canny(image, 80, 200)
    cv.namedWindow("edges", cv.WINDOW_NORMAL)
    cv.resizeWindow("edges", 500, 500)
    cv.imshow('edges', edges)
    return edges
```

<img width="365" alt="edge" src="https://user-images.githubusercontent.com/10435564/171062892-ef81a3a7-3f75-40ea-a3b9-82cbaad45355.png">


**Masking using HSV Channel**

```bash
  def red_white_masking(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    lower_y = np.array([10, 130, 120], np.uint8)
    upper_y = np.array([40, 255, 255], np.uint8)
    mask_y = cv.inRange(hsv, lower_y, upper_y)
    cv.namedWindow("mask_y", cv.WINDOW_NORMAL)
    cv.resizeWindow("mask_y", 500, 500)
    cv.imshow('mask_y', mask_y)
    
    lower_w = np.array([0, 0, 212], np.uint8)
    upper_w = np.array([170, 200, 255], np.uint8)
    mask_w = cv.inRange(hsv, lower_w, upper_w)
    cv.namedWindow("mask_w", cv.WINDOW_NORMAL)
    cv.resizeWindow("mask_w", 500, 500)
    cv.imshow('mask_w', mask_w)
    
    mask = cv.bitwise_or(mask_w, mask_y)
    cv.namedWindow("mask", cv.WINDOW_NORMAL)
    cv.resizeWindow("mask", 500, 500)
    cv.imshow('mask', mask)
    
    masked_bgr = cv.bitwise_and(image, image, mask=mask)
    cv.namedWindow("masked_bgr", cv.WINDOW_NORMAL)
    cv.resizeWindow("masked_bgr", 500, 500)
    cv.imshow('masked_bgr', masked_bgr)
    
    return masked_bgr
```
<img width="365" alt="hsv" src="https://user-images.githubusercontent.com/10435564/171062846-43d61dd6-5d9b-4c88-be7d-4a1bb8d91afd.png">


**Image Filtering**

```bash
def filtered(image):
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    filtered_image = cv.filter2D(image, -1, kernel)
    cv.namedWindow("filtered_image", cv.WINDOW_NORMAL)
    cv.resizeWindow("filtered_image", 500, 500)
    cv.imshow('filtered_image', filtered_image)
    return filtered_image
```
<img width="362" alt="filtering" src="https://user-images.githubusercontent.com/10435564/171062870-80320bf1-2a20-49fa-ba71-cbe25cbdb733.png">

**Average slope intercept**

```bash
  def average_slope_intercept(image, lines):
    left_fit = []    # list for all multiple lines found in left lane
    right_fit = []   # list for all multiple lines found in right lane
    global l
    global r
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    if left_fit != []:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = create_coordinates(image, left_fit_average)
        l = left_fit_average
    else:
        left_line = create_coordinates(image, l)
    if right_fit != []:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = create_coordinates(image, right_fit_average)
        r = right_fit_average
    else:
        right_line = create_coordinates(image, r)
    return np.array([left_line, right_line])
```
<img width="365" alt="mask" src="https://user-images.githubusercontent.com/10435564/171062941-65b2e05e-0b0d-49fa-9e67-bef81875e795.png">

## Contributors

- Rahul Arepaka
- Avula Vijaya Koushik
- Cherukuru Swaapnika Chowdary
- Lohit Garje
- Naga Tharun Makkena

## References

 - [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
 - [Building a lane detection system](https://medium.com/analytics-vidhya/building-a-lane-detection-system-f7a727c6694)
 - [Advanced Lane Detection](https://kushalbkusram.medium.com/advanced-lane-detection-fd39572cfe91)
 - [Road Lane Detection using OpenCV (Hough Lines Transform Explained)](https://medium.com/mlearning-ai/road-lane-detection-using-opencv-hough-lines-transform-explained-a6c8cfc03f68)

## Feedback

If you have any feedback, please reach out to us at rahul.arepaka@gmail.com

## License

[MIT](https://choosealicense.com/licenses/mit/)
