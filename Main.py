import MyRegression as mr
import ActiveLearing_DiffQ as adq
import ActiveLearing_DiffP as adp
file_path = 'Data/20343.csv'

# the evaluation curve of linear regression using matrix
mr.MyRegerssion(file_path)
# the evaluation curve of linear regression using gradient descent
mr.MyRegerssion(file_path, "GD")

# the evaluation learning curve of different query algorithms
adq.ActiveLearningDiffQuery(file_path)
# the evaluation learning curve of different predictive algorithms
adp.ActiveLearningDiffPredict(file_path)


