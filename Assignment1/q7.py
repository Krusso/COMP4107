import numpy as np
import matplotlib.pyplot as plt
import time

ml_dir = './ml-latest-small'
ml_100k_dir='./ml-100k'

def mean_absolute_error(y_true_table, y_pred_table, avg_user_rating):
    sumN = 0
    N = 0
    r = len(y_true_table)
    for i in range(r):
        c = len(y_true_table[i])
        for j in range(c):
            if y_true_table[i][j] != 0:
                N += 1
                sumN += abs(y_true_table[i][j] - (y_pred_table[i][j]))
    print("N is,", N)
    return sumN/N

def read(filename):
    """
        input: filename -> a string representing the path to find the data file
        return: a 2D array, where each row consists of a list of tuples (movieid, rating)
    """
    ratings = np.zeros((943, 1682))
    tuples = []
    total = 0
    with open(filename) as data_file:
        for row in data_file:
            data = row.split('\t')
            userid = int(data[0])-1
            movieid = int(data[1])-1
            tuples.append((userid, movieid))
            total = total + 1
            ratings[userid][movieid] = float(data[2])
    return ratings, tuples, total


def show_graph(results):
    for result in results:
        x = [i[0] for i in result[1]]
        y = [j[1] for j in result[1]]
        plt.plot(x, y, label=result[0])
    plt.xlabel('Folding-in model size')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()


def show_graph_put(results):
    for result in results:
        x = [i[0] for i in result[1]]
        y = [j[1] for j in result[1]]
        plt.plot(x, y, label=result[0])
    plt.xlabel('Basis size')
    plt.ylabel('Throughput (predictions/sec)')
    plt.legend()
    plt.show()
k = 14
ratings, tuples, total = read(ml_100k_dir+'/u.data')

movie_averages = np.true_divide(ratings.sum(0), (ratings != 0).sum(0))
user_averages = np.true_divide(ratings.sum(1), (ratings != 0).sum(1))

# !!!! EXPERIMENTAL !!!
#
#
# for m in range(len(movie_averages)):
#     for u in range(len(user_averages)):
#         if ratings[u][m] != 0:
#             ratings[u][m] = ratings[u][m] - movie_averages[m]

for m in range(len(movie_averages)):
    for u in range(len(user_averages)):
        if ratings[u][m] != 0:
            ratings[u][m] = ratings[u][m] - user_averages[u]
# !!!! END EXPERIMENTAL !!!

resultGraph = []
timeGraph = []
for split_percentage in list([0.2, 0.5, 0.8]):
    subset = np.random.choice(total, int(total * split_percentage), replace=False)
    train_set = []
    train_data = np.zeros((943, 1682))
    test_set = []
    test_data = np.zeros((943, 1682))
    for d in range(total):
        t = tuples[d]
        if d in subset:
            train_set.append(d)
            train_data[t[0]][t[1]] = ratings[t[0]][t[1]]
        else:
            test_set.append(d)
            test_data[t[0]][t[1]] = ratings[t[0]][t[1]]

    results = []
    times = []
    for b in list([600, 650, 700, 750, 800, 850, 900, 943]):
        threshold_size = b  # from research paper
        start = time.time()
        u, s, v = np.linalg.svd(train_data[:threshold_size])
        uk = u[:,:k]
        sk = np.diag(s[:k])
        vk = v[:k]
        # fold in
        for i in range(threshold_size, 943):
            nu = np.array(train_data[i])
            P = np.dot(np.dot(nu, vk.T), np.linalg.inv(sk))
            uk = np.vstack([uk, P])

        print("Starting to make predictions")
        # make predictions
        m = np.dot(uk, np.sqrt(sk).T)
        n = np.dot(np.sqrt(sk), vk)
        pred_table = np.dot(m,n)

        print("Pred", pred_table.shape)
        print("Test", len(test_set))
        print("Train", len(train_set))
        mae = mean_absolute_error(test_data[:943], pred_table, user_averages)
        end = time.time()
        print(mae)
        print(uk.shape)
        results.append((b, mae))
        times.append((b, len(test_set) / (end - start)))

    resultGraph.append((split_percentage, results))
    timeGraph.append((split_percentage, times))

show_graph(resultGraph)
show_graph_put(timeGraph)
