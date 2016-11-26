import numpy as np
from numpy.random import uniform, randint
np.random.seed(555)


class Arm:
    def __init__(self, feature_dim, content_id, alpha=0.01):
        self.content_id = content_id
        self.alpha = alpha  # regularization parameter
        self.norm_mean = np.zeros((1, feature_dim))  # b
        # Covariance matrix:
        # Store diagonal components only to increase memory efficiency
        self.cov_matrix = np.ones((1, feature_dim))

        self.win_rate = 0
        self.win = 0
        self.lose = 0

    def update(self, features, is_click):
        features = features.reshape((1,features.shape[0]))
        self.cov_matrix += np.diag(features.T.dot(features))
        self.norm_mean += is_click*features
        if is_click:
            self.win+=1
        else:
            self.lose+=1
        self.win_rate = self.win/(self.win + self.lose)

    def predict(self, features):
        features = features.reshape((features.shape[0],1))
        # Since the covariance matrix preserves only the diagonal components,
        # it suffices to take the inverse matrix
        theta = (1/self.cov_matrix)*self.norm_mean   # [1, feature_dim]
        # Again, the inverse matrix of the covariance matrix
        # is ​​computed by taking reciprocal
        return theta.dot(features) + \
            self.alpha*np.sqrt((features.T*(1/self.cov_matrix)).dot(features))

    def print_result(self):
        print('content_id:{}, total_num:{}, win_rate:{}'.format(\
                self.content_id, self.win+self.lose, self.win_rate))


class Viewer:
    def __init__(self, gender='man'):
        self.gender = gender

    def view(self, content_id):
        if self.gender == 'man':
            # Men are easy to click on ads with id 5 or less
            if content_id<6:
                return True if uniform(0, 1.0) > 0.3 else False
            else:
                return True if uniform(0, 1.0) > 0.7 else False
        else:
            # Women are easy to click on ads with id 6 or higher
            if content_id > 5:
                return True if uniform(0, 1.0) > 0.3 else False
            else:
                return True if uniform(0, 1.0) > 0.7 else False


class Rulet:
    def __init__(self, feature_dim):
        self.arms = {}
        self.feature_dim = feature_dim

    def generate_arm(self, content_id):
        if content_id not in self.arms:
            self.arms[content_id] = Arm(self.feature_dim, content_id)
        return self.arms[content_id]

    def generate_features(self):
        viewer = Viewer(self.generate_gender())
        features = np.array([1,0]) if viewer.gender=='man' else np.array([0,1])
        content_id = self.generate_content()
        return features, viewer.view(content_id), self.generate_arm(content_id)

    def generate_content(self):
        return randint(1, 10)

    def generate_gender(self):
        return 'man' if uniform(0, 1.0) > 0.5 else 'women'


if __name__=='__main__':
    '''Context is for men and women only
    Men are easy to click on ads with id 5 or less
    Women are easy to click on ads with id 6 or higher
    '''
    alpha = 0.0001  # regularization parameter
    feature_dim = 2
    num_of_views = 10000
    rulet = Rulet(feature_dim)

    for step in range(num_of_views):
        features, is_clicked, arm = rulet.generate_features()
        arm.update(features, is_clicked)
        # Confirmation of prediction accuracy when the number of data is small
        if step<2000:
            arm.print_result()

    print('print result======')
    for content_id, arm in rulet.arms.items():
        arm.print_result()
        print('Click rate when men browse:' + str(arm.predict(np.array([1,0]))) )
        print('Click rate when women browse:'+ str(arm.predict(np.array([0,1]))) )
