import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


data_path = 'ml-1m/ratings.dat'
n_users = 610
n_movies = 9742


def load_rating_data(data_path, n_users, n_movies):
    data = np.zeros([n_users, n_movies], dtype=np.float32)
    movie_id_mapping = {}
    movie_n_rating = defaultdict(int)
    with open(data_path, 'r') as file:
        for line in file.readlines()[1:]:
            user_id, movie_id, rating, _ = line.split("::")
            user_id = int(user_id) - 1
            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)
            rating = int(rating)
            data[user_id, movie_id_mapping[movie_id]] = rating
            if rating > 0:
                movie_n_rating[movie_id] += 1
    return data, movie_n_rating, movie_id_mapping


data, movie_n_rating, movie_id_mapping = load_rating_data(data_path, n_users, n_movies)


def display_distribution(data):
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print(f'Number of rating {int(value)}: {count}')


movie_id_most, n_rating_most = sorted(movie_n_rating.items(), key=lambda d: d[1], reverse=True)[0]

# we split most rated movie in Y and keep others in X
X_raw = np.delete(data, movie_id_mapping[movie_id_most], axis=1)
Y_raw = data[:, movie_id_mapping[movie_id_most]]

X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]
Y1 = Y_raw[Y_raw > 0]

recommend = 3
Y[Y <= recommend] = 0
Y[Y > recommend] = 1
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()

# we split x and y to two dataset of trained and test to see how good we can predict the result
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)
prediction_prob = clf.predict_proba(X_test)
prediction = clf.predict(X_test)
accuracy = clf.score(X_test, Y_test)

print("---------------display distribution-----------------")
display_distribution(data)
print("-------------------most rating----------------------")
print(f'Movie ID {movie_id_most} has {n_rating_most} ratings.')
print("----------------------------------------------------")
print('Shape of data:', data.shape)
print("----------------------------------------------------")
print('Shape of X:', X.shape)
print("----------------------------------------------------")
print('Shape of Y:', Y1.shape)
print("------------display distribution of Y---------------")
display_distribution(Y1)
print("----------------------------------------------------")
print(f'{n_pos} positive samples and {n_neg} negative samples.')
print("------------------train test split------------------")
print(len(Y_train), len(Y_test))
print("---------------predicted probabilities--------------")
print(prediction_prob[0:10])
print("----------------------------------------------------")
print(prediction[:10])
print("---------------classification accuracy--------------")
print(f'The accuracy is: {accuracy * 100:.1f}%')
print("----------------------------------------------------")

