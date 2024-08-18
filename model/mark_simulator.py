import numpy as np


def simulate(timestamps, timestamp_dims, beta, dims, v, num_sentiments=5):
	sentiments = np.linspace(-1, 1, num=num_sentiments)
	size = timestamps.size
	marks = np.empty(size)
	marks[0] = np.random.choice(sentiments)
	timestamps_beta = np.empty((dims, timestamps.size))
	intensity = np.empty(size)
	for i in range(dims):
		timestamps_beta[i] = beta[:, i][timestamp_dims]
	for i in range(1, size):
		dim = timestamp_dims[i]
		time_diff = timestamps[i] - timestamps[:i]
		intensity[i] = np.sum(timestamps_beta[dim, :i] * marks[:i] * np.exp(-v * time_diff))
		p = np.exp(sentiments * intensity[i]) / np.sum(np.exp(sentiments * intensity[i]))
		marks[i] = np.random.choice(sentiments, p=p)
	return marks


def simulate_exo_history(history, endo_mask, beta, dims, v, num_sentiments=3):
	timestamps = np.array([h[1] for h in history])
	timestamp_dims = np.array([h[0] for h in history])
	sentiments = np.linspace(-1, 1, num=num_sentiments)
	size = timestamps.size
	marks = np.empty(size)
	marks[0] = np.random.choice(sentiments)
	timestamps_beta = np.empty((dims, timestamps.size))
	intensity = np.empty(size)
	for i in range(dims):
		timestamps_beta[i] = beta[:, i][timestamp_dims]
	for i in range(1, size):
		if not endo_mask[i]:
			marks[i] = np.random.choice(sentiments)
		else:
			dim = timestamp_dims[i]
			time_diff = timestamps[i] - timestamps[:i]
			intensity[i] = np.sum(timestamps_beta[dim, :i] * marks[:i] * np.exp(-v * time_diff))
			p = np.exp(sentiments * intensity[i]) / np.sum(np.exp(sentiments * intensity[i]))
			marks[i] = np.random.choice(sentiments, p=p)
	return marks


def simulate_exo(timestamps, timestamp_dims, endo_mask, beta, dims, v, num_sentiments=3):
	sentiments = np.linspace(-1, 1, num=num_sentiments)
	size = timestamps.size
	marks = np.empty(size)
	marks[0] = np.random.choice(sentiments)
	timestamps_beta = np.empty((dims, timestamps.size))
	intensity = np.empty(size)
	for i in range(dims):
		timestamps_beta[i] = beta[:, i][timestamp_dims]
	for i in range(1, size):
		if not endo_mask[i]:
			marks[i] = np.random.choice(sentiments)
		else:
			dim = timestamp_dims[i]
			time_diff = timestamps[i] - timestamps[:i]
			intensity[i] = np.sum(timestamps_beta[dim, :i] * marks[:i] * np.exp(-v * time_diff))
			p = np.exp(sentiments * intensity[i]) / np.sum(np.exp(sentiments * intensity[i]))
			marks[i] = np.random.choice(sentiments, p=p)
	return marks
