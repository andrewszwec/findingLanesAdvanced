import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points
nx = 8#TODO: enter the number of inside corners in x
ny = 6#TODO: enter the number of inside corners in y

# Make a list of calibration images
fname = 'calibration_test.png'
img = cv2.imread(fname)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)

##################################################
# CALIBRATE CAMERA AND UNDISTORT
##################################################
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]

# Read in an image
img = cv2.imread('test_image.png', -1)

# TODO: Write a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

undistorted = cal_undistort(img, objpoints, imgpoints)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

##################################################
# USEFUL CODE
##################################################
# Examples of Useful Code

# Converting an image, imported by cv2 or the glob API, to grayscale:

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Note: If you are reading in an image using mpimg.imread() this will read in an RGB image and you should convert to grayscale using cv2.COLOR_RGB2GRAY, but if you are using cv2.imread() or the glob API, as happens in this video example, this will read in a BGR image and you should convert to grayscale using cv2.COLOR_BGR2GRAY. We'll learn more about color conversions later on in this lesson, but please keep this in mind as you write your own code and look at code examples.

# Finding chessboard corners (for an 8x6 board):

ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
# Drawing detected corners on an image:

img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
# Camera calibration, given object points, image points, and the shape of the grayscale image:

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[:-1], None, None)
# Undistorting a test image:

dst = cv2.undistort(img, mtx, dist, None, mtx)


##################################################
# COMPUTING IMAGE TRANSFORMATION - TOP DOWN VIEW
##################################################
# Compute the perspective transform, M, given source and destination points:
M = cv2.getPerspectiveTransform(src, dst)
# Compute the inverse perspective transform:
Minv = cv2.getPerspectiveTransform(dst, src)
# Warp an image using the perspective transform, M:
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)


##################################################
# UNDISTORTING AND TRANSFORMING AN IMAGE
##################################################
# Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                     [img_size[0]-offset, img_size[1]-offset],
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M


##################################################
# My Attempt
##################################################
# Undistort
undist = cv2.undistort(img, mtx, dist, None, mtx)
# Convert to grayscale
#gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
    plt.imshow(gray)

    # points
    # src = np.float32([[465.32049561,768.59197998],[459.21780396,650.59698486],[586.10174561,747.33929443],[583.51916504,636.49847412]])
    # dst = np.float32([[465.32049561,768.59197998],[465.32049561,650.59698486],[583.51916504,768.59197998],[583.51916504,650.59698486]])
    src = np.float32([[465,769],[1075,658],[437,114],[1104,225]])
    dst = np.float32([[465,769],[1075,769],[465,114],[1075,114]])
    # Compute the perspective transform, M, given source and destination points:
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (gray.shape[1], gray.shape[0])
    # Warp an image using the perspective transform, M:
    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)





a = [ 436.53210449,
     573.60150146,
     691.33123779,
     793.93548584,
     884.17254639,
     966.10614014,
    1038.02270508,
    1104.04943848,
     441.4074707 ,
     575.54962158,
     691.69976807,
     792.48120117,
     881.59362793,
     962.27056885,
    1034.0065918 ,
    1097.83227539,
     447.1946106 ,
     577.43804932,
     692.35894775,
     791.17095947,
     878.69287109,
     957.01165771,
    1026.68725586,
    1092.3515625 ,
     453.24633789,
     580.82794189,
     692.29119873,
     790.29003906,
     875.60778809,
     953.77172852,
    1023.14202881,
    1086.        ,
     459.21780396,
     583.51916504,
     692.82592773,
     788.68707275,
     874.1350708 ,
     949.93499756,
    1018.21520996,
    1079.96557617,
     465.32049561,
     586.10174561,
     693.05212402,
     786.72167969,
     871.26940918,
     946.49298096,
    1014.79870605,
    1075.49963379]

b = [114.34142303,
137.80006409,
156.4487915 ,
174.432724  ,
190.40956116,
203.37982178,
215.59300232,
224.65359497,
256.5692749 ,
268.57223511,
279.86419678,
289.6335144 ,
297.71035767,
304.30029297,
312.3732605 ,
319.41101074,
393.76025391,
395.54278564,
397.63717651,
400.25158691,
401.76379395,
403.72796631,
405.16540527,
407.13397217,
525.33154297,
518.73968506,
513.14050293,
507.56420898,
502.75720215,
499.41033936,
495.13476562,
493.5       ,
650.59698486,
636.49847412,
623.27960205,
611.75292969,
600.49786377,
592.41223145,
583.84246826,
575.98638916,
768.59197998,
747.33929443,
728.57141113,
710.85076904,
695.89093018,
681.24176025,
669.43341064,
658.01580811]

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)


ax.scatter(a,b, label='True Position')
for x, y in zip(a, b):
    text = str(round(x)) + ', ' + str(round(y))
    ax.text(x, y, text)

plt.show()

##################################################
# SOBEL - USEFUL CODE
##################################################
# Examples of Useful Code
#
# You need to pass a single color channel to the cv2.Sobel() function, so first convert it to grayscale:
#
# gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# Note: Make sure you use the correct grayscale conversion depending on how you've read in your images. Use cv2.COLOR_RGB2GRAY if you've read in an image using mpimg.imread(). Use cv2.COLOR_BGR2GRAY if you've read in an image using cv2.imread().
#
# Calculate the derivative in the x direction (the 1, 0 at the end denotes x direction):
#
# sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
# Calculate the derivative in the y direction (the 0, 1 at the end denotes y direction):
#
# sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
# Calculate the absolute value of the x derivative:
#
# abs_sobelx = np.absolute(sobelx)
# Convert the absolute value image to 8-bit:
#
# scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
# Note: It's not entirely necessary to convert to 8-bit (range from 0 to 255) but in practice, it can be useful in the event that you've written a function to apply a particular threshold, and you want it to work the same on input images of different scales, like jpg vs. png. You could just as well choose a different standard range of values, like 0 to 1 etc.
#
# Create a binary threshold to select pixels based on gradient strength:
#
# thresh_min = 20
# thresh_max = 100
# sxbinary = np.zeros_like(scaled_sobel)
# sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
# plt.imshow(sxbinary, cmap='gray')
# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output



##################################################
# Magnititude Thresholding - USEFUL CODE
##################################################
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


##################################################
# Gradient Direction for Feature Detection - USEFUL CODE
##################################################
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output



##################################################
# Project 4 - Challenge
##################################################
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    return dir_binary

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(0, 255))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))
Try different combinations and see what you get.

For example, here is a selection for pixels where both the x and y gradients meet the threshold criteria, or the gradient magnitude and direction are both within their threshold values.

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
