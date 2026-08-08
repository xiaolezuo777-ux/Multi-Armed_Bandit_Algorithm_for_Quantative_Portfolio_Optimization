import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# New base directory after extraction
base_dir = 'C:/Users/zuoxi/Desktop/ml-1m'

# Paths to the specific files
movies_path = os.path.join(base_dir, 'movies.dat')
ratings_path = os.path.join(base_dir, 'ratings.dat')

# Load data, specifying the encoding as Latin-1
movies = pd.read_csv(movies_path, delimiter='::', engine='python', names=['MovieID', 'Title', 'Genres'], header=None, encoding='latin-1')
ratings = pd.read_csv(ratings_path, delimiter='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'], header=None, encoding='latin-1')

# Prepare genres
movies['Genres'] = movies['Genres'].str.split('|')
movies = movies.explode('Genres')

# Merge ratings with movies
merged_data = pd.merge(ratings, movies, on='MovieID')

# calculate average ratings for each genre
average_rewards = merged_data.groupby('Genres')['Rating'].mean().values

class ThompsonSampling:
    def __init__(self, n_arms, B):
        self.n_arms = n_arms
        self.B = B
        self.rewards_sum = np.zeros(n_arms) # cumulative rewards for each arm
        self.rewards_counts = np.zeros(n_arms) # chosen times for each arm
        self.mean_rewards = np.zeros(n_arms)

    def select_arm(self):
        """ sampling by Gaussian distribution and choose the best arm """
        sampled_values =[]
        for arm in range(self.n_arms):
            variance = (self.B ** 2) / (4 * self.rewards_counts[arm])
            sampled_value = np.random.normal(self.mean_rewards[arm], np.sqrt(variance))
            sampled_values.append(sampled_value)

        return np.argmax(sampled_values)

    def update(self, chosen_arm, reward):
        """ update times"""
        self.rewards_sum[chosen_arm] += reward
        self.rewards_counts[chosen_arm] += 1
        self.mean_rewards[chosen_arm] = self.rewards_sum[chosen_arm] / self.rewards_counts[chosen_arm]
        return np.argmax(self.rewards_counts)

genres = merged_data['Genres'].unique()
genre_dict = {genre: merged_data[merged_data['Genres'] == genre] for genre in merged_data['Genres'].unique()}
horizon = 100000

def simulate_thompson_sampling(data, genres, genre_dict, num_experiments, horizon, B):
    """running for one time on experiments"""
    n_arms = len(genres)
    cumulative_regret = np.zeros((num_experiments, horizon))

    for exp in range(num_experiments):
        # Renew the ThompsonSampling
        ts = ThompsonSampling(n_arms, B)
        # Pre-sample 100000(horizon) ratings for each genre and cache them to reduce repeated sampling
        pre_sampled_ratings = {genre: genre_dict[genre]['Rating'].sample(horizon, replace=True).values for genre in
                               genres}

        # Pre-sample 1 rating for each arm initially
        for arm in range(n_arms):
            random_rating = np.random.choice(pre_sampled_ratings[genres[arm]])
            best_arm = ts.update(arm, random_rating)
            if arm == 0:
                cumulative_regret[exp, arm] = average_rewards[best_arm] - random_rating
            else:
                cumulative_regret[exp, arm] = cumulative_regret[exp, arm-1] + average_rewards[best_arm] - random_rating


        # start from k+1
        for t in range(n_arms, horizon):
            chosen_arm = ts.select_arm()
            random_reward = np.random.choice(pre_sampled_ratings[genres[chosen_arm]])
            best_arm = ts.update(chosen_arm, random_reward)

            cumulative_regret[exp, t] = cumulative_regret[exp, t - 1] + average_rewards[best_arm] - average_rewards[chosen_arm]

    return cumulative_regret

'''
num_experiments = 10
regret_results = simulate_thompson_sampling(merged_data, genres, genre_dict, num_experiments, horizon, 4)
plt.figure(figsize=(10, 6))
for exp in range(num_experiments):
    plt.plot(np.arange(1, horizon + 1), regret_results[exp, :], label=f'Experiment {exp + 1}')
plt.title('Thompson Sampling Algorithm')
plt.xlabel('Round')
plt.ylabel('Regret')
plt.legend()
plt.show()
'''

'''
num_experiments = 100
horizon = 100000
regret_results = simulate_thompson_sampling(merged_data, genres, genre_dict, num_experiments, horizon, 4)
average_regret = np.mean(regret_results, axis=0)
std_dev_regret = np.std(regret_results, axis=0)

plt.figure(figsize=(10, 6))
plt.errorbar(np.arange(1, horizon+1), average_regret, yerr=std_dev_regret, label='Average Regret with Error Bar')

plt.title('TS Algorithm (100 Experiments)')
plt.xlabel('Round')
plt.ylabel('Average Regret')
plt.legend()
plt.show()
'''
'''
num_experiments = 100
horizons = [500, 5000, 50000, 500000, 5000000]
for horizon in horizons:
    regret_results = simulate_thompson_sampling(merged_data, genres, genre_dict, num_experiments, horizon, 4)
    print(regret_results[0][horizon-1])
    plt.plot(np.arange(1, horizon + 1), np.mean(regret_results, axis=0), label=f'n = {horizon}')
    plt.title('TS Algorithm - Average Regret for Different Horizons')
    plt.xlabel('Round')
    plt.ylabel('Average Regret')
    plt.legend()
    plt.show()
'''

# Explore-then-Commit Algorithm
class ETCBandit:
    def __init__(self, k, exploration_phase):
        self.k = k  # number of arms
        self.exploration_phase = exploration_phase  # number of times to explore each arm
        self.counts = np.zeros(k)  # counts of selections for each arm
        self.sums = np.zeros(k)  # sum of rewards for each arm

    def pull(self, arm, reward):
        """Update the counts and sums after pulling an arm """
        self.counts[arm] += 1
        self.sums[arm] += reward

    def get_best_arm(self):
        """ Return the arm with the highest average reward after exploration """
        averages = self.sums / self.counts
        return np.argmax(averages), averages  # return the biggest arm(index) of averages


def simulate_etc(data, genres, horizon, num_experiments, experiment_phase_frequency):
    exploration_phase = int(horizon * experiment_phase_frequency)
    k = len(genres)
    m = exploration_phase // k
    cumulative_regret = np.zeros((num_experiments, horizon))
    for exp in range(num_experiments):
        bandit = ETCBandit(k, exploration_phase)

        # Initialize an array to store cumulative rewards at each step
        total_actual_rewards = np.zeros(horizon)

        # Exploration phase
        for arm in range(k):
            arm_genre = genres[arm]
            filtered_data = data[data['Genres'] == arm_genre]
            for t in range(m):
                reward = np.random.choice(filtered_data['Rating'])
                bandit.pull(arm, reward)
                if t < horizon:
                    total_actual_rewards[t] = reward if t == 0 else total_actual_rewards[t - 1] + reward

        # print(bandit.counts)
        # print(bandit.sums)

        # Calculate the best arm
        best_arm, average_reward = bandit.get_best_arm()
        best_genre = genres[best_arm]
        # print(reward[reward['Genres']==best_genre] * horizon)

        # Calculate the maximum possible reward with the optimal strategy
        best_rating = bandit.sums[best_arm] / bandit.counts[best_arm]
        optimal_rewards = best_rating * np.arange(1, horizon + 1)  # Cumulative optimal rewards over time
        # print(optimal_rewards)

        # Commit phase
        filtered_data = data[data['Genres'] == genres[best_arm]]
        for t in range(m * k, horizon):
            reward = np.random.choice(filtered_data['Rating'])
            total_actual_rewards[t] = total_actual_rewards[t - 1] + reward
        # print(total_actual_rewards)

        # Calculate cumulative regret at each time step
        cumulative_regret[exp, :] = optimal_rewards - total_actual_rewards

    return cumulative_regret


# Step 1: dealing the information with movies data into a new list
# Loading the Movies Data
def load_data_movies(file_path):
    # step 1: Initialize an empty list to store the processed movie data
    data = []
    # step 2: Open the file with 'latin-1' encoding
    with open(file_path, 'r', encoding='latin-1') as file:
        # step 3: Read each line in the file one by one
        for line in file:
            # step 4: Remove the newline character at the end of each line
            line = line.strip('\n')
            # step 5: Split the line into parts using '::' as the delimiter(as readme file said)
            line = line.split('::')
            # step 6: Split the genres string into individual genres using '|' as a delimiter(as readme file said)
            genres = line[2].split('|')
            # step 7: Initialize an empty list to store the integer odes for each genre
            genres_int = []
            # step 8: Iterate over each genre in the list of genres. Each genre is uniquely identified by an integer form 0 to 1
            for genre in genres:
                if genre == 'Action':
                    genres_int.append(0)
                if genre == 'Adventure':
                    genres_int.append(1)
                if genre == 'Animation':
                    genres_int.append(2)
                if genre == "Children's":
                    genres_int.append(3)
                if genre == 'Comedy':
                    genres_int.append(4)
                if genre == 'Crime':
                    genres_int.append(5)
                if genre == 'Documentary':
                    genres_int.append(6)
                if genre == 'Drama':
                    genres_int.append(7)
                if genre == 'Fantasy':
                    genres_int.append(8)
                if genre == 'Film-Noir':
                    genres_int.append(9)
                if genre == 'Horror':
                    genres_int.append(10)
                if genre == 'Musical':
                    genres_int.append(11)
                if genre == 'Mystery':
                    genres_int.append(12)
                if genre == 'Romance':
                    genres_int.append(13)
                if genre == 'Sci-Fi':
                    genres_int.append(14)
                if genre == 'Thriller':
                    genres_int.append(15)
                if genre == 'War':
                    genres_int.append(16)
                if genre == 'Western':
                    genres_int.append(17)
            # step 9: Replace the genres in the original line with their integer codes
            line[2] = genres_int
            # step 10: Append the modified line to the main data list
            data.append(line)
    return data


# Step 2: rating data
def load_data_ratings(file_path):
    data = []
    with open(file_path, 'r', encoding='latin-1') as file:
        for line in file:
            line = line.strip('\n')
            line = line.split('::')
            data.append(line)
    return data


# Step 3: combine movie ID with rating
def classification(data_movies, data_ratings):
    # step 1: Creat 18 empt lists for eacj genre
    data = []
    for i in range(18):
        data.append([])
    # step 2: Iterate over each movie in the movies data
    for movie in data_movies:
        # step 3: Iterate over each rating in the ratings data
        for rating_line in data_ratings:
            # step 4: Match movie ID in ratings:
            if int(movie[0]) == int(rating_line[1]):
                for genre in movie[2]:
                    data[genre].append(int(rating_line[2]))
    return data


# Step 4: Intializes UCB algorithm
class UpperConfidenceBound:
    def __init__(self, num_actions, horizon, B):
        self.num_actions = num_actions
        self.horizon = horizon
        self.B = B
        self.action_counts = np.zeros(num_actions)
        self.cumulative_rewards = np.zeros(num_actions)
        self.round = 0

    # Step 5: define UCB algorithm calculation and asymptotically optimal UCB algorithms
    def select_action(self, l):
        average_rewards = np.zeros(self.num_actions)
        for i in range(self.num_actions):
            if self.action_counts[i] == 0:
                average_rewards[i] = 5
            else:
                average_rewards[i] = self.cumulative_rewards[i] / self.action_counts[i] + math.sqrt(
                    l * math.log(self.horizon) / self.action_counts[i]) * self.B / 2
        action = np.argmax(average_rewards)
        self.round += 1
        return action

    def asymptotical_UCB_action(self, l):
        average_rewards = np.zeros(self.num_actions)
        for i in range(self.num_actions):
            if self.action_counts[i] == 0:
                average_rewards[i] = 5
            else:
                average_rewards[i] = self.cumulative_rewards[i] / self.action_counts[i] + math.sqrt(
                    l * math.log(1 + (i + 1) * math.pow(math.log(i + 1), 2) / self.action_counts[i]) * self.B / 2)
        action = np.argmax(average_rewards)
        self.round += 1
        return action

    # Step 6: update results
    def update(self, action, reward):
        self.action_counts[action] += 1
        self.cumulative_rewards[action] += reward


# Step 7
def run_experiment(data, num_experiments, horizon, l):
    results = np.zeros((num_experiments, horizon))
    expect_rating = 0
    for genre in data:
        if sum(genre) / len(genre) > expect_rating:
            expect_rating = sum(genre) / len(genre)
    B = 4  # rewards (i.e., ratings) can be in the interval 1-5 (stars),B should be set as 4
    for exp in range(num_experiments):
        np.random.shuffle(data)
        num_actions = len(data)
        etc_algorithm = UpperConfidenceBound(num_actions, horizon, B)
        for t in range(horizon):
            selected_action = etc_algorithm.select_action(l)
            reward = data[selected_action][np.random.randint(0, len(data[selected_action]))]
            etc_algorithm.update(selected_action, reward)
            cumulative_regret = expect_rating * (t + 1) - np.sum(etc_algorithm.cumulative_rewards)
            results[exp, t] = cumulative_regret
    return results


def run_experiment_asymptotically(data, num_experiments, horizon, l):
    results = np.zeros((num_experiments, horizon))
    expect_rating = 0
    for genre in data:
        if sum(genre) / len(genre) > expect_rating:
            expect_rating = sum(genre) / len(genre)
    B = 4
    for exp in range(num_experiments):
        np.random.shuffle(data)
        num_actions = len(data)
        aUCB_algorithm = UpperConfidenceBound(num_actions, horizon, B)
        for t in range(horizon):
            selected_action = aUCB_algorithm.asymptotical_UCB_action(l)
            reward = data[selected_action][np.random.randint(0, len(data[selected_action]))]
            aUCB_algorithm.update(selected_action, reward)
            cumulative_regret = expect_rating * (t + 1) - np.sum(aUCB_algorithm.cumulative_rewards)
            results[exp, t] = cumulative_regret
    return results


# input the data
if __name__ == "__main__":
    movie_path = "C:/Users/zuoxi/Desktop/movies.dat"
    rating_path = "C:/Users/zuoxi/Desktop/ratings.dat"
    data_movies = load_data_movies(movie_path)
    data_ratings = load_data_ratings(rating_path)
    data = classification(data_movies, data_ratings)

horizon = 1000000
num_experiments = 100
regret_results_ts = simulate_thompson_sampling(merged_data, genres, genre_dict, num_experiments, horizon, 4)
average_regret_ts = np.mean(regret_results_ts, axis=0)
std_dev_regret_ts = np.std(regret_results_ts, axis=0)

regret_results_etc = simulate_etc(merged_data, genres, horizon, num_experiments, 0.1)
average_regret_etc = np.mean(regret_results_etc, axis=0)
std_dev_regret_etc = np.std(regret_results_etc, axis=0)

regret_results_ucb = run_experiment(data, num_experiments, horizon, 4)
average_regret_ucb = np.mean(regret_results_ucb, axis=0)
std_dev_regret_ucb = np.std(regret_results_ucb, axis=0)

regret_results_as_optimal_ucb = run_experiment_asymptotically(data, num_experiments, horizon, 4)
average_regret_as_optimal_ucb = np.mean(regret_results_as_optimal_ucb, axis=0)
std_dev_regret_as_optimal_ucb = np.std(regret_results_as_optimal_ucb, axis=0)


plt.figure(figsize=(10, 6))
plt.errorbar(np.arange(1, horizon+1), average_regret_etc, yerr=std_dev_regret_etc, label='ETC', color='blue', linestyle='-')
plt.errorbar(np.arange(1, horizon+1), average_regret_ucb, yerr=std_dev_regret_ucb, label='UCB', color='green', linestyle='--')
plt.errorbar(np.arange(1, horizon+1), average_regret_as_optimal_ucb, yerr=std_dev_regret_as_optimal_ucb, label='asymptotically optimal UCB', color='orange', linestyle='-')
plt.errorbar(np.arange(1, horizon+1), average_regret_ts, yerr=std_dev_regret_ts, label='TS', color='red', linestyle='--')

plt.title('Comparison of Algorithms (100 Experiments)')
plt.xlabel('Round')
plt.ylabel('Average Regret')
plt.legend(loc='best')
plt.show()
