import numpy as np 
import csv
import os
import pprint
# from sklearn.metrics import mean_absolute_error

ml_dir = './ml-latest-small'
# MAX_RATINGS = 100000

pp = pprint.PrettyPrinter(indent = 4)

#A map of userid to an array of ratings
user_rates = {} 
movies = {} #map of movieid to index

k=14 #

#ratings.csv contains approx 100000 ratings with each user having more than 20 ratings
#userid, movieid , rating, timestamp
with open(ml_dir + '/ratings.csv') as ratings_file:
    csv_reader = csv.reader(ratings_file, delimiter=',')
    next(csv_reader) #skip the first line as those are just the labels
    for row in csv_reader:
        userid = row[0]
        if userid not in user_rates:
            user_rates[userid]=[]
        user_rates[userid].append(row[1:])

with open(ml_dir + '/movies.csv', encoding='utf8') as movie_file:
    csv_reader = csv.reader(movie_file, delimiter = ',')
    next(csv_reader)
    index = 0
    for row in csv_reader:
        # print(row)
        movieid = row[0]
        if movieid not in movies:
            movies[movieid] = index
            index += 1

def convertToMatrix(ur):
    #Convert the dictionary of user rates which is a mapping of userid to [movieid, rating, timestamp]
    numUsers = len(ur)
    numMovies = len(movies)
    print("Making a %d x %d user-movie matrix" % (numUsers, numMovies))
    mat = []
    for _, value in ur.items():
        usrRatings = [0 for i in range(numMovies)]
        for data in value:
            movieId = data[0]
            rating = data[1]
            usrRatings[movies[movieId]] = float(rating)
        mat.append(usrRatings)
    return mat

matrix = convertToMatrix(user_rates)
matrix = np.matrix(matrix)
print(matrix.shape)

def split(mat, x=0.8):
    numTrainExamples = int(len(mat) * x)
    # print(numTrainExamples)
    training = matrix[:numTrainExamples]
    testing = matrix[numTrainExamples:]
    return training, testing

train_set, test_set = split(matrix)
print(train_set.shape)
u, s, vt = np.linalg.svd(train_set, full_matrices=False)
print(u.shape, s.shape, vt.shape)

uk = u[:,:k]
sk = np.diag(s[:k])
vkt = vt[:k]

print(uk.shape)
print(sk.shape)
print(vkt.shape)

usv = uk*sk*vkt

print(usv.shape)
print(usv)

basis = []

for bSize in range(600,950,50):
    print(bSize)