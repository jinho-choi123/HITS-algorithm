import numpy as np

def hit_algorithm(A: np.ndarray, n_iter: int = 1000):
    '''
    Given adjacent matrix an adjacent matrix, calculate the authority score and hub score for each node.
    It iters the HIT algorithm n_iter times.

    :param A: Adjacent matrix for the graph
    :returns: authority score and hub scores
    '''

    # check the validity of Adjacent matrix A's shape
    N, N_ = A.shape
    assert N == N_

    A_T = A.transpose()

    # calculate the matmul of A and A_T
    A_A_T = np.matmul(A, A_T)

    # initialize hubs_score and authorities_score
    h_score = np.ones(N)
    a_score = np.ones(N)

    for _ in range(n_iter):
        a_score = np.matmul(A_T, h_score)
        h_score = np.matmul(A_A_T, h_score)

        # apply normalization
        a_norm = np.linalg.norm(a_score)
        h_norm = np.linalg.norm(h_score)

        assert a_norm > 1e-5
        assert h_norm > 1e-5

        a_score = a_score / a_norm
        h_score = h_score / h_norm

    # return the a_score and h_score
    return (a_score, h_score)

def main():
    # generate adjacent matrix for the graph
    A = np.array(
        [
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [1, 1, 1, 0, 1, 0]
    ])

    (a_score, h_score) = hit_algorithm(A, 10000)

    print(f"a_score: {a_score}")
    print(f"h_score: {h_score}")
    return

if __name__ == "__main__":
    main()
