import numpy as np
import sklearn
from tick.hawkes import SimuPoissonProcess


def kernel (x,y,b):
	return np.exp (-b * (x-y))


def drawExpRV (param, rng):
	return rng.exponential (scale=param)


# simulate poisson distribution events
def simu_poiss(mu, T):
	poi = SimuPoissonProcess(mu, end_time=T, verbose=False)
	poi.simulate()
	timestamps = poi.timestamps
	formated_timestamps = [(i, t, 1) for i in range(mu.size) for t in timestamps[i]]
	return formated_timestamps


# Simulate timestamps cascade with presence of exogenous events
def multivariate_exo(mu_exo, alpha, omega, T, numEvents=None, checkStability=False, seed=None):
	dim = alpha.shape[0]
	mu = np.random.uniform(0, 0.02, dim)
	history, T = multivariate(mu, alpha, omega, T, numEvents, dim, checkStability, seed)
	exos = simu_poiss(mu_exo, T)
	history = history + exos
	history.sort(key=lambda x: x[1])
	return history


# Simulate Hawkes Cascade
def multivariate(mu, alpha, omega, T, numEvents, dim, checkStability, seed):
	# make mu small!
	prng = sklearn.utils.check_random_state (seed)
	nTotal = 0
	history = list ()
	# Initialization
	if numEvents is None:
		nExpected = np.iinfo (np.int32).max
	else:
		nExpected = numEvents
	s = 0.0

	if checkStability:
		w,v = np.linalg.eig (alpha)
		maxEig = np.amax (np.abs(w))
		if maxEig >= 1:
			print("(WARNING) Unstable ... max eigen value is: {0}".format (maxEig))

	Istar = np.sum(mu)
	s += drawExpRV (1. / Istar, prng)

	if s <=T and nTotal < nExpected:
		# attribute (weighted random sample, since sum(mu)==Istar)
		n0 = int(prng.choice(np.arange(dim), 1, p=(mu / Istar)))
		history.append((n0, s, 0))
		nTotal += 1

	# value of \lambda(t_k) where k is most recent event
	# starts with just the base rate
	lastrates = mu.copy()

	decIstar = False
	while nTotal < nExpected:
		if len(history) == 0:
			return history, 0.
		uj, tj = int (history[-1][0]), history[-1][1]

		if decIstar:
			# if last event was rejected, decrease Istar
			Istar = np.sum(rates)
			decIstar = False
		else:
			# otherwise, we just had an event, so recalc Istar (inclusive of last event)
			Istar = np.sum(lastrates) + alpha[uj,:].sum()

		s += drawExpRV (1. / Istar, prng)
		if s > T:
			break

		# calc rates at time s (use trick to take advantage of rates at last event)
		rates = mu + kernel (s, tj, omega) * (alpha[uj, :] + lastrates - mu)

		# attribution/rejection test
		# handle attribution and thinning in one step as weighted random sample
		diff = Istar - np.sum(rates)
		n0 = int (prng.choice(np.arange(dim+1), 1, p=(np.append(rates, diff) / Istar)))

		if n0 < dim:
			history.append((n0, s, 0))
			# update lastrates
			lastrates = rates.copy()
			nTotal += 1
		else:
			decIstar = True

	T = history[-1][1]
	return history, T
