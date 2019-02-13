import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

# Read image of Ima and resize by 1/4
path = ''
#path = 'C:/Users/Katherine/Documents/School/2019 Spring/CSC 391'
img = cv2.imread(path + '2031a382-71dc-4fd0-9ef7-72d011a9eebd.jpg')
small = cv2.resize(img, None, fx=1, fy=1)
cv2.imshow('Scaling - Linear Interpolation', img)
cv2.waitKey(0)
blur = cv2.GaussianBlur(img,(27,27),0)
cv2.imshow('Gaussian', blur)
cv2.waitKey(0)
# kernel_sharpen_2 = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
# output_2 = cv2.filter2D(img, -1, kernel_sharpen_2)
# cv2.imshow('edge detection', output_2)
# cv2.waitKey(0)
#
# # Create noisy image as g = f + sigma * noise
# # with noise scaled by sigma = .2 max(f)/max(noise)
# noise = np.random.randn(small.shape[0], small.shape[1])
# smallNoisy = np.zeros(small.shape, np.float64)
# sigma = 0.2 * small.max()/noise.max()
# # Color images need noise added to all channels
# if len(small.shape) == 2:
#     smallNoisy = small + sigma * noise
# else:
#     smallNoisy[:, :, 0] = small[:, :, 0] + sigma * noise
#     smallNoisy[:, :, 1] = small[:, :, 1] + sigma * noise
#     smallNoisy[:, :, 2] = small[:, :, 2] + sigma * noise
#
# # Calculate the index of the middle column in the image
# col = int(smallNoisy.shape[1]/2)
# # Obtain the image data for this column
# colData = smallNoisy[0:smallNoisy.shape[0], col, 0]
#
# # Plot the column data as a stem plot of xvalues vs colData
# xvalues = np.linspace(0, len(colData) - 1, len(colData))
# markerline, stemlines, baseline = plt.stem(xvalues, colData, 'b')
# plt.setp(markerline, 'markerfacecolor', 'b')
# plt.setp(baseline, 'color', 'r', 'linewidth', 0.5)
# plt.show()
#
# # Add a line to the noisy image, save to file, and display
# #cv2.line(smallNoisy, (col,0), (col,smallNoisy.shape[0]),(0,0,255), 2)
# #cv2.imwrite(path + 'Ima-noisy.jpg', smallNoisy)
# #cv2.imshow('image', img)
# #cv2.waitKey(0)
#
# # Create a 1-D filter of length 3: [1/3, 1/3, 1/3] and apply to column data
# windowLen = 27
# w = np.ones(windowLen, 'd')  # vector of ones
# w = w / w.sum()
# y = np.convolve(w, colData, mode='valid')
#
# # Plot the filtered column data as a stem plot
# xvalues = np.linspace(0, len(y)-1, len(y))
# markerline, stemlines, baseline = plt.stem(xvalues, y, 'g')
# plt.setp(markerline, 'markerfacecolor', 'g')
# plt.setp(baseline, 'color','r', 'linewidth', 0.5)
# plt.show()
#
# # Create a 2-D box filter of size 3 x 3 and scale so that sum adds up to 1
# w = np.ones((windowLen, windowLen), np.float32)
# w = w / w.sum()
# denoised = np.zeros(smallNoisy.shape, np.float64)  # array for denoised image
# # Apply the filter to each channel
# denoised[:, :, 0] = cv2.filter2D(smallNoisy[:, :, 0], -1, w)
# denoised[:, :, 1] = cv2.filter2D(smallNoisy[:, :, 1], -1, w)
# denoised[:, :, 2] = cv2.filter2D(smallNoisy[:, :, 2], -1, w)
#
# print(denoised.min())
# print(denoised.max())
#
# denoised_copy = denoised
# # 1st option: Scaled the denoised image back to uint8 using astype()
# #denoised = (denoised + denoised.min()) / smallNoisy.max()  # preserve range and scale to [0, 1] if min() is negative
# denoised = denoised / smallNoisy.max()  # scale to [min/max, 1]
# denoised = denoised * 255 # scale to [min/max*255, 255]
# #cv2.imshow('denoised', denoised.astype(np.uint8))  # convert to uint8
# #cv2.waitKey(0)
# # 2nd option: Scale the denoised image using skimage.img_as_ubyte()
# #denoised = denoised_copy
# #denoised = denoised / smallNoisy.max()
# #cv2.imshow('denoise2', ski.img_as_ubyte(denoised))
# cv2.waitKey(0)
# #cv2.imwrite(path + 'Ima-Denoised.jpg', denoised)
# gaussian=np.array([[1,1,1],[1,1,1],[1,1,1]])/9.0
# kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# kernel_sharpen_2 = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
# kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],
#                              [-1,2,2,2,-1],
#                              [-1,2,8,2,-1],
#                              [-1,2,2,2,-1],
#                              [-1,-1,-1,-1,-1]]) / 8.0
#
# # applying different kernels to the input image
# output_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
# output_2 = cv2.filter2D(img, -1, kernel_sharpen_2)
# output_3 = cv2.filter2D(img, -1, kernel_sharpen_3)
# output_4 = cv2.filter2D(img, -1, gaussian)
# blur = cv2.GaussianBlur(img,(5,5),0)
# cv2.imshow('Sharpening', output_1)
# cv2.imshow('edge detection', output_2)
# cv2.imshow('Edge Enhancement', output_3)
# cv2.imshow('Gaussian', blur)

cv2.waitKey(0)