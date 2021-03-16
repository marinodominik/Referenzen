import numpy as np


class Problem2:

    def euclidean_square_dist(self, features1, features2):
        """ Computes pairwise Euclidean square distance for all pairs.
        Args:
            features1: (128, m) numpy array, descriptors of first image         (128, 123)
            features2: (128, n) numpy array, descriptors of second image        (128, 140)
        Returns:
            distances: (n, m) numpy array, pairwise distances
        """
        _, m = features1.shape
        _, n = features2.shape
        distances = np.zeros((n, m))  # (140, 123)

        for i in range(m):
            q = features1[:, i]
            for j in range(n):
                p = features2[:, j]
                d = ((q - p) ** 2).sum()
                distances[j, i] = d  # die distanz von jedem feature1 zu feature 2 steht in der i-ten spalte

        return distances

    def find_matches(self, p1, p2, distances):
        """ Find pairs of corresponding interest points given the
        distance matrix.
        Args:
            p1: (m, 2) numpy array, keypoint coordinates in first image     #(123, 2)
            p2: (n, 2) numpy array, keypoint coordinates in second image    #(140, 2)
            distances: (n, m) numpy array, pairwise distance matrix         #(140, 123)
        Returns:
            pairs: (min(n,m), 4) numpy array s.t. each row holds        (123, 4)
                the coordinates of an interest point in p1 and p2.
        """
        m, _ = p1.shape
        n, _ = p2.shape
        pairs = np.zeros((min(n, m), 4))  # (123, 4)

        for i in range(m):
            """ look for minimum value in distances and get idx"""
            row, column = np.where(distances == np.amin(distances.flatten()))
            """ look the found idx in p1 and p2 and set it to pairs"""
            pairs[i, :2] = p1[column, :]
            pairs[i, 2:4] = p2[row, :]
            """ block row and column of the distances and set it to infinity"""
            distances[row, :] = np.inf
            distances[:, column] = np.inf

        return pairs

    def pick_samples(self, p1, p2, k):
        """ Randomly select k corresponding point pairs.
        Args:
            p1: (n, 2) numpy array, given points in first image
            p2: (m, 2) numpy array, given points in second image
            k:  number of pairs to select
        Returns:
            sample1: (k, 2) numpy array, selected k pairs in left image
            sample2: (k, 2) numpy array, selected k pairs in right image
        """
        index = np.random.choice(p1.shape[0], k, replace=False)

        sample1 = p1[index]
        sample2 = p2[index]

        return sample1, sample2

    def condition_points(self, points):
        """ Conditioning: Normalization of coordinates for numeric stability
        by substracting the mean and dividing by half of the component-wise
        maximum absolute value.
        Further, turns coordinates into homogeneous coordinates.
        Args:
            points: (l, 2) numpy array containing unnormailzed cartesian coordinates.
        Returns:
            ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
            T: (3, 3) numpy array, transformation matrix for conditioning
        """
        l, _ = points.shape

        def scale_and_shift(p):
            return np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)

        p = scale_and_shift(points)

        s = 0.5 * np.max(abs(p))
        t = np.mean(points, axis=0)

        T = np.diag([1 / s, 1 / s, 1])
        T[0, 2] = - t[0] / s
        T[1, 2] = - t[1] / s
        ps = T @ np.hstack((points, np.ones((l, 1)))).T

        return ps.T, T

    def compute_homography(self, p1, p2, T1, T2):
        """ Estimate homography matrix from point correspondences of conditioned coordinates.
        Both returned matrices shoul be normalized so that the bottom right value equals 1.
        You may use np.linalg.svd for this function.
        Args:
            p1: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img1
            p2: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img2
            T1: (3,3) numpy array, conditioning matrix for p1
            T2: (3,3) numpy array, conditioning matrix for p2

        Returns:
            H: (3, 3) numpy array, homography matrix with respect to unconditioned coordinates
            HC: (3, 3) numpy array, homography matrix with respect to the conditioned coordinates
        """

        coordinate_points = len(p1)
        A = np.zeros((2 * coordinate_points, 9))

        x = p1[:, 0]
        y = p1[:, 1]
        x_ = p2[:, 0]
        y_ = p2[:, 1]

        for i in range(coordinate_points):
            A[2 * i] = [0, 0, 0, x[i], y[i], 1, - x[i] * y_[i], - y[i] * y_[i], - y_[i]]
            A[2 * i + 1] = [-x[i], -y[i], -1, 0, 0, 0, x[i] * x_[i], y[i] * x_[i], x_[i]]

        _, _, V = np.linalg.svd(A)
        HC = V[8].reshape((3, 3))
        H = np.dot(np.dot(np.linalg.inv(T2), HC), T1)

        return H / H[2, 2], HC / HC[2, 2]  # normalize that bottom right value equals 1

    def transform_pts(self, p, H):
        """ Transform p through the homography matrix H.
        Args:
            p: (l, 2) numpy array, interest points
            H: (3, 3) numpy array, homography matrix

        Returns:
            points: (l, 2) numpy array, transformed points
        """
        points = np.dot(H, np.hstack((p, np.ones((len(p), 1)))).T)
        points = points[0:2, :] / points[2, :]

        return points[0:2, :].T

    def compute_homography_distance(self, H, p1, p2):
        """ Computes the pairwise symmetric homography distance.
        Args:
            H: (3, 3) numpy array, homography matrix
            p1: (l, 2) numpy array, interest points in img1
            p2: (l, 2) numpy array, interest points in img2

        Returns:
            dist: (l, ) numpy array containing the distances
        """
        l, _ = p1.shape

        problem = Problem2()

        x1transform = problem.transform_pts(p1, H)
        x2transform = problem.transform_pts(p2, np.linalg.inv(H))

        return np.sum(np.sqrt((x1transform - p2) ** 2 + (p1 - x2transform) ** 2), axis=1)

    def find_inliers(self, pairs, dist, threshold):
        """ Return and count inliers based on the homography distance.
        Args:
            pairs: (l, 4) numpy array containing keypoint pairs
            dist: (l, ) numpy array, homography distances for k points
            threshold: inlier detection threshold

        Returns:
            N: number of inliers
            inliers: (N, 4)
        """
        return len(pairs[dist < threshold]), pairs[dist < threshold]


    def ransac_iters(self, p, k, z):
        """ Computes the required number of iterations for RANSAC.
        Args:
            p: probability that any given correspondence is valid   0.35
            k: number of pairs                                      4
            z: total probability of success after all iterations    0.99

        Returns:
            minimum number of required iterations
        """
        return int(np.round(np.log(1 - z) / np.log(1 - (p ** k))))


    def ransac(self, pairs, n_iters, k, threshold):
        """ RANSAC algorithm.
        Args:
            pairs: (l, 4) numpy array containing matched keypoint pairs
            n_iters: number of ransac iterations
            threshold: inlier detection threshold

        Returns:
            H: (3, 3) numpy array, best homography observed during RANSAC
            max_inliers: number of inliers N
            inliers: (N, 4) numpy array containing the coordinates of the inliers
        """
        H_return = None
        n_max = None
        inliers = None

        for i in range(n_iters):
            s1, s2 = self.pick_samples(pairs[:, 0:2], pairs[:, 2:4], k)
            p1, T1 = self.condition_points(s1)
            p2, T2 = self.condition_points(s2)
            H, HC = self.compute_homography(p1, p2, T1, T2)
            dist = self.compute_homography_distance(H, pairs[:, 0:2], pairs[:, 2:4])
            n, inl = self.find_inliers(pairs, dist, threshold)

            if (i == 0):
                H_return = H
                n_max = n
                inliers = inl
            elif (n_max < n):
                H_return = H
                n_max = n
                inliers = inl

        return H_return, n_max, inliers

    def recompute_homography(self, inliers):
        """ Recomputes the homography matrix based on all inliers.
        Args:
            inliers: (N, 4) numpy array containing coordinate pairs of the inlier points

        Returns:
            H: (3, 3) numpy array, recomputed homography matrix
        """
        u, T1 = self.condition_points(inliers[:, :2])
        u_, T2 = self.condition_points(inliers[:, 2:4])
        H, _ = self.compute_homography(u, u_, T1, T2)
        return H