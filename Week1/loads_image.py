from astropy.io import fits
import matplotlib.pyplot as plt

hdulist = fits.open('image0.fits')
data = hdulist[0].data

# Plot the 2D array
plt.imshow(data, cmap=plt.cm.viridis)
plt.xlabel('x-pixels (RA)')
plt.ylabel('y-pixels (Dec)')
plt.colorbar()
plt.show()



from astropy.io import fits
import numpy as np

def load_fits(filename):
  hdulist = fits.open(filename)
  data = hdulist[0].data

  arg_max = np.argmax(data)  
  max_pos = np.unravel_index(arg_max, data.shape)
  
  return max_pos