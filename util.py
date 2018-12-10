
def subtract_vectors(v1, v2):
    return [a - b for a, b in zip(v1, v2)]

def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))
