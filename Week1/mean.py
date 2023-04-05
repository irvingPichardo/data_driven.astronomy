from statistics import mean
fluxes = [23.3, 42.1, 2.0, -3.2, 55.6]
m = mean(fluxes)
print(m)


fluxes = [23.3, 42.1, 2.0, -3.2, 55.6]
m = sum(fluxes)/len(fluxes)
print(m)


# Write your calculate_mean function here.
def calculate_mean(data):
  mean = sum(data)/len(data)
  return mean

# You can use this to test your function.
# Any code inside this `if` statement will be ignored by the automarker.
if __name__ == '__main__':
  # Run your `calculate_mean` function with examples:
  mean = calculate_mean([1,2.2,0.3,3.4,7.9])
  print(mean)
  

import numpy as np
fluxes = np.array([23.3, 42.1, 2.0, -3.2, 55.6])
m = np.mean(fluxes)
print(m)


import numpy as np
fluxes = np.array([23.3, 42.1, 2.0, -3.2, 55.6])
print(np.size(fluxes)) # length of array
print(np.std(fluxes))  # standard deviation



import numpy as np
# Write your mean_datasets function here
def mean_datasets(filenames):
  n = len(filenames)
  if n > 0:
    data = np.loadtxt(filenames[0], delimiter=',')
    for i in range(1,n):
      data += np.loadtxt(filenames[i], delimiter=',')
    
    # Mean across all files:
    data_mean = data/n
     
    return np.round(data_mean, 1)


# You can use this to test your function.
# Any code inside this `if` statement will be ignored by the automarker.
if __name__ == '__main__':
  # Run your function with the first example from the question:
  print(mean_datasets(['data1.csv', 'data2.csv', 'data3.csv']))

  # Run your function with the second example from the question:
  print(mean_datasets(['data4.csv', 'data5.csv', 'data6.csv']))



import numpy as np

a = np.array([[1,2,3], [4,5,6]])  # 2x3 array
# Print first row of a:
print(a[0,:])
# Print second column of a:
print(a[:,1])
