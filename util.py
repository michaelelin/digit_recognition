
def subtract_vectors(v1, v2):
	return { feature: v1[feature] - v2[feature] for feature in v1.keys() }

def dot_product(v1, v2):
	sum = 0
	for feature in v1.keys():
		sum += v1[feature] * v2[feature]
	return sum
