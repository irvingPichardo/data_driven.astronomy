import psycopg2

# Establish the connection
conn = psycopg2.connect(dbname='db', user='grok')
cursor = conn.cursor()

# Execute an SQL query and receive the output
cursor.execute('SELECT 2 + 3;')
records = cursor.fetchall()

print(records)



import psycopg2

conn = psycopg2.connect(dbname='db', user='grok')
cursor = conn.cursor()
cursor.execute('SELECT * FROM Star')
records = cursor.fetchall()

t_eff = []
for row in records:
  t_eff.append(row[1])

print(t_eff)



import psycopg2
import numpy as np

conn = psycopg2.connect(dbname='db', user='grok')
cursor = conn.cursor()

cursor.execute('SELECT radius FROM Star;')

records = cursor.fetchall()
array = np.array(records)

print(array.shape)
print(array.mean())
print(array.std())


import numpy as np
import psycopg2

def column_stats(table, col):
  conn = psycopg2.connect(dbname='db', user='grok')
  cursor = conn.cursor()

  query = 'SELECT ' + col + ' FROM ' + table + ';'
  cursor.execute(query)
  column = np.array(cursor.fetchall())
  return np.mean(column), np.median(column)


# Write your query function here
import numpy as np

def query(f_name):
  data = np.loadtxt(f_name, delimiter=',', usecols=(0, 2))
  return data[data[:, 1]>1, :]


# You can use this to test your code
# Everything inside this if-statement will be ignored by the automarker
if __name__ == '__main__':
  # Compare your function output to the SQL query
  result = query('stars.csv')
  
  

# Write your query function here

import numpy as np

def query(fname_1, fname_2):
  stars = np.loadtxt(fname_1, delimiter=',', usecols=(0, 2))
  planets = np.loadtxt(fname_2, delimiter=',', usecols=(0, 5))

  f_stars = stars[stars[:,1]>1, :]                
  s_stars = f_stars[np.argsort(f_stars[:, 1]), :] 
 
  final = np.zeros((1, 1))
  for i in range(s_stars.shape[0]):
    kep_id = s_stars[i, 0]
    s_radius = s_stars[i, 1]

    matching_planets = planets[np.where(planets[:, 0] == kep_id), 1].T
    final = np.concatenate((final, matching_planets/s_radius))

  return np.sort(final[1:], axis = 0)

# You can use this to test your code
# Everything inside this if-statement will be ignored by the automarker
if __name__ == '__main__':
  # Compare your function output to the SQL query
  result = query('stars.csv', 'planets.csv')