
from celery import shared_task
import numpy as np
import random
from datetime import datetime

@shared_task
def recommend_task(data):
    k = 3
    lr = 0.01
    reg = 0.01
    epochs = 200
    verbose = True
    print_results = False
    for_test = False
    test_user_num = 100
    test_problem_num = 100

    start_time = datetime.now()
    print("cf start - " + str(start_time))


    traslator = Translator(data, for_test, test_user_num, test_problem_num)
    R, solved_matrix = traslator.translateFromJson()

    factorizer = MatrixFactorization(
        R, k=k, learning_rate=lr, reg_param=reg, epochs=epochs, verbose=verbose
    )
    factorizer.fit()
    predicted_matrix = factorizer.final_matrix()

    if print_results:
        factorizer.print_results()

    end_time = datetime.now()
    print("cf end - " + str(end_time))
    print("total time - " + str((end_time-start_time).total_seconds()))

    return traslator.translateToJson(predicted_matrix, solved_matrix)



class Translator:
    def __init__(self, data, for_test, test_user_num, test_problem_num):
        self.data = data
        self.for_test = for_test
        self.test_user_num = test_user_num
        self.test_problem_num = test_problem_num
        self.json_maxUserId = "maxUserId"
        self.json_maxProblemId = "maxProblemId"
        self.json_data = "data"
        self.json_userId = "userId"
        self.json_problemId = "problemId"
        self.json_problemStatus = "problemStatus"
        self.json_unsolvedCnt = "unsolvedCnt"
        self.json_success = "SUCCESS"
        self.json_process = "PROCESS"
        self.json_unknown = "unknown"

    # 테스트 데이터 생성
    def testDataMaker(self):
        content = dict()
        content[self.json_data] = []
        content[self.json_maxUserId] = self.test_user_num
        content[self.json_maxProblemId] = self.test_problem_num

        for userId in range(self.test_user_num):
            for problemId in range(self.test_problem_num):
                jform = dict()
                jform[self.json_userId] = userId
                jform[self.json_problemId] = problemId
                jform[self.json_problemStatus] = self.randomProblemStatus()
                jform[self.json_unsolvedCnt] = self.randomUnsolvedCnt()

                content[self.json_data].append(jform)

        return content

    # (테스트 데이터용) 맞췄는지, 풀이중인지 랜덤 반환
    def randomProblemStatus(self):
        ranVal = random.randrange(2)
        if ranVal == 0:
            return self.json_success
        else:
            return self.json_process

    # (테스트 데이터용) 틀린 횟수 랜덤 반환
    def randomUnsolvedCnt(self):
        ranVal = random.randrange(10)
        return ranVal

    # Json 데이터를 행렬로 변환
    def translateFromJson(self):
        if self.for_test:
            content = self.testDataMaker()
        else:
            content = self.data

        user_count = content[self.json_maxUserId]
        problem_count = content[self.json_maxProblemId]

        solve_status_lst = np.array(content[self.json_data])
        wrong_cnt_matrix = np.zeros(shape=(user_count, problem_count))
        solved_matrix = np.zeros((user_count, problem_count), dtype=bool)

        for ss in solve_status_lst:
            wrong_cnt_matrix[ss[self.json_userId] - 1][ss[self.json_problemId] - 1] = ss[self.json_unsolvedCnt]
            solved_matrix[ss[self.json_userId] - 1][ss[self.json_problemId] - 1] = True

        return wrong_cnt_matrix, solved_matrix

    # 행렬을 Json 데이터로 변환
    def translateToJson(self, wrong_cnt_matrix, solved_matrix):
        content = {self.json_data: []}

        r, c = wrong_cnt_matrix.shape
        for userId in range(r):
            recommend_count = 0
            skipped = []
            for problemId in range(c):
                if solved_matrix[userId][problemId]:
                    skipped.append((problemId, wrong_cnt_matrix[userId][problemId]))
                    continue
                if wrong_cnt_matrix[userId][problemId] > 0:
                    json_form = dict()
                    json_form[self.json_userId] = userId+1
                    json_form[self.json_problemId] = problemId+1
                    content[self.json_data].append(json_form)
                    recommend_count += 1
            
            if recommend_count < 3:
                skipped.sort(key=lambda x: x[1], reverse = True)

                for i in range(3-recommend_count):
                    problemId, _ = skipped[i]

                    json_form = dict()
                    json_form[self.json_userId] = userId+1
                    json_form[self.json_problemId] = problemId+1
                    content[self.json_data].append(json_form)

        return content

class MatrixFactorization:
    def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):
        """
        R: rating matrix
        k: latent parameter
        learning_rate: alpha on weight update
        reg_param: beta on weight update
        epochs: training epochs
        verbose: print status
        """

        self._R = R
        self._num_users, self._num_items = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose

    def fit(self):
        """
        self._b
        - global bias: input R에서 평가가 매겨진 rating의 평균값을 global bias로 사용
        - 정규화 기능. 최종 rating에 음수가 들어가는 것 대신 latent feature에 음수가 포함되도록 해줌.
        """
        start_time = datetime.now()

        # init latent features
        self._P = np.random.normal(size=(self._num_users, self._k))
        self._Q = np.random.normal(size=(self._num_items, self._k))

        # init biases
        self._b_P = np.zeros(self._num_users)
        self._b_Q = np.zeros(self._num_items)
        self._b = np.mean(self._R[np.where(self._R != 0)])

        # train while epochs
        self._training_process = []
        for epoch in range(self._epochs):

            # rating이 존재하는 index를 기준으로 training
            for i in range(self._num_users):
                for j in range(self._num_items):
                    if self._R[i, j] > 0:
                        self.gradient_descent(i, j, self._R[i, j])
            cost = self.cost()
            self._training_process.append((epoch, cost))

            # print status
            if self._verbose == True:
                cur_time = datetime.now()
                if epoch == 0:
                    time_diff = cur_time - start_time
                else:
                    time_diff = cur_time - prev_time
                print("Iteration: %d ; cost = %.4f; time diff = %s" % (epoch + 1, cost, str(time_diff.total_seconds())))
                prev_time = cur_time

    def cost(self):
        xi, yi = self._R.nonzero()
        predicted = self.get_complete_matrix()
        cost = 0
        for x, y in zip(xi, yi):
            cost += pow(self._R[x, y] - predicted[x, y], 2)
        return np.sqrt(cost) / len(xi)

    def gradient(self, error, i, j):
        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
        return dp, dq

    def gradient_descent(self, i, j, rating):
        # get error
        prediction = self.get_prediction(i, j)
        error = rating - prediction

        # update biases
        self._b_P[i] += self._learning_rate * (error - self._reg_param * self._b_P[i])
        self._b_Q[j] += self._learning_rate * (error - self._reg_param * self._b_Q[j])

        # update latent feature
        dp, dq = self.gradient(error, i, j)
        self._P[i, :] += self._learning_rate * dp
        self._Q[j, :] += self._learning_rate * dq

    def get_prediction(self, i, j):
        return (
            self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)
        )

    def get_complete_matrix(self):
        return (
            self._b
            + self._b_P[:, np.newaxis]
            + self._b_Q[np.newaxis :,]
            + self._P.dot(self._Q.T)
        )

    def rounding(self, matrix):
        return np.round(matrix, 0)

    def limiting(self, matrix):
        matrix = np.where(matrix < 0, 0, matrix)
        return matrix

    def final_matrix(self):
        return self.limiting(self.rounding(self.get_complete_matrix()))

    def print_results(self):
        print()
        print("Origin R Matrix:")
        print(self._R)
        print()
        print("Predicted R matrix:")
        print(self.get_complete_matrix())
        print()
        print("Rounding(Predicted R matrix):")
        print(self.rounding(self.get_complete_matrix()))
        print()
        print("Limiting(Predicted R matrix):")
        print(self.final_matrix())
        print()
        print("Final RMSE:")
        print(self._training_process[self._epochs - 1][1])
