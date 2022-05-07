from skimage.io import imread
import pandas
from sklearn.cluster import KMeans
import numpy as np
from skimage import img_as_float
import pylab
import math
import matplotlib.pyplot as plt

#1
img = img_as_float(imread('parrots.jpg'))
pylab.imshow(img)
# plt.show()

#2
pixels = pandas.DataFrame(np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2])), columns=['R', 'G', 'B'])

#3
model = KMeans(init='k-means++', random_state=241)
model.fit(pixels.loc[:,'R':'B'])
pixels['Cluster'] = model.predict(pixels.loc[:,'R':'B'])

means = pixels.groupby('Cluster').mean().values
mean_pixels = [means[x] for x in pixels['Cluster']]
mean_image = np.reshape(mean_pixels, (img.shape[0],img.shape[1],img.shape[2]))

medians = pixels.groupby('Cluster').median().values
median_pixels = [medians[x] for x in pixels['Cluster']]
median_image = np.reshape(median_pixels, (img.shape[0],img.shape[1],img.shape[2]))
# pylab.imshow(mean_image)
# plt.show()
#
# pylab.imshow(median_image)
# plt.show()

#4
psnr_median = 10 * math.log10(float(1) / np.mean((img - median_image) ** 2))
psnr_mean = 10 * math.log10(float(1) / np.mean((img - mean_image) ** 2))

# print(psnr_median)
# print(psnr_mean)

#5
k = 0
for n_clusters in range(1, 21):
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241)
    model.fit(pixels.loc[:, 'R':'B'])
    pixels['Cluster'] = model.predict(pixels.loc[:, 'R':'B'])

    means = pixels.groupby('Cluster').mean().values
    mean_pixels = [means[x] for x in pixels['Cluster']]
    mean_image = np.reshape(mean_pixels, (img.shape[0], img.shape[1], img.shape[2]))

    medians = pixels.groupby('Cluster').median().values
    median_pixels = [medians[x] for x in pixels['Cluster']]
    median_image = np.reshape(median_pixels, (img.shape[0], img.shape[1], img.shape[2]))

    psnr_median = 10 * math.log10(float(1) / np.mean((img - median_image) ** 2))
    psnr_mean = 10 * math.log10(float(1) / np.mean((img - mean_image) ** 2))

    if psnr_median > 20 or psnr_mean > 20:
        k = n_clusters
        break

print(k)
