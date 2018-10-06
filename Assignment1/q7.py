import numpy as np 
import csv
import os
import pprint
# from sklearn.metrics import mean_absolute_error

ml_dir = './ml-latest-small'
ml_100k_dir='./ml-100k'

pp = pprint.PrettyPrinter(indent = 4)

#A map of userid to an array of ratings
user_rates = {} 
movies = {} #map of movieid to index
k=14 

def movie_rating_avg(table, movieIndex):
    s = 0.0
    num_ratings = 0
    for r in range(len(table)):
        if table[r][movieIndex] != 0:
            s += table[r][movieIndex]
            num_ratings += 1

    return float(s/num_ratings)

def mean_absolute_error(y_true_table, y_pred_table, avg_user_rating):
    sumN = 0
    N = 0
    r = len(y_true_table)
    for i in range(r):
        c = len(y_true_table[i])
        # row_average = sum([i for i in y_pred_table[i]])/c
        # print(row_average)
        for j in range(c):
            if y_true_table[i][j] != 0:
                N += 1
                # row_average = movie_rating_avg(y_true_table, j)
                sumN += abs(y_true_table[i][j] - (avg_user_rating[i] + y_pred_table[i][j]))

    mae = sumN/N
    return mae

# mae = mean_absolute_error(train_set, pred_table)
# print(mae)

# for basis_size in range(400, 601, 50):
#     A = matrix[]

def read(filename):
    """
        input: filename -> a string representing the path to find the data file
        return: a 2D array, where each row consists of a list of tuples (movieid, rating)
    """
    user_ratings = [[] for i in range(943)]
    with open(filename) as data_file:
        for row in data_file:
            data = row.split('\t')
            userid = int(data[0])-1
            movieid = int(data[1])-1
            rating = (movieid, float(data[2]))
            user_ratings[userid].append(rating)
    return user_ratings

def split_data(ur, split_percentage=0.2):
    """
        input: ur -> the user rating that was returned from calling read()
               split_percentage -> the percentage we are splitting the data training:test

        return: 2d matrices representing the training and test set
    """
    np.random.shuffle(ur)

    num_items = 1682
    train_set = [[0 for i in range(num_items)] for j in range(len(ur))]
    test_set = [[0 for i in range(num_items)] for j in range(len(ur))]
    average_rating = []
    for userIndex in range(len(ur)):
        #randomy choose split_percentage of ratings for each user and put them
        #in the trainset, while the rest should be placed in the test set
        user_rating = ur[userIndex] #this is a list of tuples (movieid, rating)
        num_ratings = len(ur[userIndex])
        training_choices = np.random.choice([i for i in range(num_ratings)], int(num_ratings*split_percentage))
        #loop from 0 - 1682, these index numbers represent the movieid, and thus A[i][j] = user_rating
        avg = 0
        N = 0
        for i in range(num_ratings):
            #we check if i is in training_choices, and if so, we set user rating here
            if i in training_choices:
                train_set[userIndex][user_rating[i][0]] = user_rating[i][1]
                avg += user_rating[i][1]
                N += 1
            else:
                test_set[userIndex][user_rating[i][0]] = user_rating[i][1]
        avg /= N
        average_rating.append(avg)
    return train_set, test_set, average_rating


k = 14
user_ratings = read(ml_100k_dir+'/u.data')
train_set, test_set, avg_user_rating = split_data(user_ratings, split_percentage=0.9)
threshold_size = 600 #from research paper

u, s, v = np.linalg.svd(train_set[:threshold_size])
uk = u[:,:k]
sk = np.diag(s[:k])
vk = v[:k]
#fold in
for i in range(threshold_size, 900):
    print(i)
    nu = np.array(train_set[i])
    P = np.dot(np.dot(nu, vk.T), np.linalg.inv(sk))
    uk = np.vstack([uk,P])

#make predictions
m = np.dot(uk, np.sqrt(sk).T)
n = np.dot(np.sqrt(sk), vk)
pred_table = np.dot(m,n)

# for i in range(len(test_set)):
#     for j in range(len(test_set[i])):
#         if test_set[i][j] != 0:
#             print("prediction for %d,%d is %d and real is %d" % (i,j ,pred_table[i][j], test_set[i][j]))

# for i in range(len(train_set)):
#     for j in range(len(train_set[i])):
#         if train_set[i][j] != 0:
#             print("prediction for %d,%d is %f and real is %f" % (i,j ,pred_table[i][j], train_set[i][j]))
           


print(pred_table.shape)
mae = mean_absolute_error(test_set[:900], pred_table, avg_user_rating)

print(mae)




print(uk.shape)