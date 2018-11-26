class FeatureVector(dict):

	def __init__(self):
		super(FeatureVector, self).__init__()

	def __sub__(self, other):
		return FeatureVector({ feature: self[feature] - other[feature] for feature in self.keys() })

	def __mul__(self, other):
		sum = 0
		for feature in self.keys():
			sum += self[feature] * other[feature]
		return sum