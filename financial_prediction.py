# we use svr model provided by sklearn to predict the S&P 500 Index 
# the S&P data is collected by finance.yahoo and time from 1996/1 to 2010/11
import csv, math, numpy
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from scipy.optimize import minimize

# extract close price from file with designed filename
def read_rawdata_csv(filename):
    trade_date = []
    close_price = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            trade_date.append(row['Date'])
            close_price.append(float(row['Close']))
    return trade_date, close_price
# transform close price into moving average
def process_rawdata(trade_date,close_price,candidate_average = [1,5,10,20,50]):
    moving_average = {}
    trade_date = trade_date[max(candidate_average):len(trade_date)]
    for i in candidate_average:
        moving_average[i] = []
    for i in range(max(candidate_average),len(close_price)):
        for j in candidate_average:
            moving_average[j].append(sum(close_price[i-j+1:i+1:1])/j)
    return trade_date, moving_average
# assemble moving average into log return
def process_return_list(trade_date,moving_average):
    return_dict = {}
    for i in moving_average.keys():
        return_dict[i] = []
    for i in range(max(moving_average.keys()),len(trade_date)):
        for j in return_dict.keys():
            return_dict[j].append(math.log(moving_average[j][i]) - math.log(moving_average[j][i-j]))
    trade_date = trade_date[max(moving_average.keys()):len(trade_date)]
    return trade_date, return_dict
# assemble log return into svr feature
def process_feature(trade_date,return_dict,candidate_average = [1,5,10,20,50],p = [10,8,6,4,4]):
    truncation_n = candidate_average[len(candidate_average)-1]*p[len(p)-1]
    trade_date = trade_date[truncation_n:len(trade_date)]
    x = numpy.zeros([len(trade_date),sum(p)])
    y = numpy.zeros([len(trade_date)])
    for i in range(0,len(trade_date)-1):
        tmp = []
        for j in range(0,len(candidate_average)):
            tmp = tmp + return_dict[candidate_average[j]][i+truncation_n-(p[j]-1)*candidate_average[j]:i+truncation_n+1:candidate_average[j]]
        x[i,:] = numpy.array(tmp)
        y[i] = return_dict[candidate_average[0]][i+truncation_n+1]
    return trade_date, x, y
# train sparse linear combination of svr
def train_svr(x_train, y_train, x_test, y_test, p_sum = [0,10,18,24,28,32]):
    svr_list = []
    for i in range(0,len(p_sum)-1):
        clf_regress = SVR(C = 0.1, epsilon = 0.0, gamma = 0.01)
        clf_regress.fit(x_train[0:x_train.shape[0],p_sum[i]:p_sum[i+1]],y_train)
        svr_list.append(clf_regress)
    s0 = numpy.zeros(len(svr_list)+1)
    s0[0] = 1
    arg = (svr_list,x_test,y_test,p_sum)
    cons = {'type':'ineq','fun':lambda s:sum(s[i] for i in range(0,len(s)-1)) - 1}
    s = minimize(min_fun,s0,arg,method='COBYLA',constraints=cons)
    return svr_list, s.x

# the minifunction for sparse linear combination
def min_fun(s, svr_list, x, y, p_sum):
    y_pre = numpy.zeros(y.shape)
    for i in range(0,len(p_sum)-1):
        clf_regress = svr_list[i]
        y_pre = y_pre + math.fabs(s[i])*clf_regress.predict(x[0:x.shape[0],p_sum[i]:p_sum[i+1]])
    y_pre = y_pre + s[len(p_sum)-1]
    turnover = []
    for i in range(0,len(y)):
        if y_pre[i]*y[i] > 0:
            turnover.append(math.fabs(y[i]))
        else:
            turnover.append(-math.fabs(y[i]))
    return numpy.mean(turnover)/numpy.std(turnover)*math.sqrt(252) + numpy.linalg.norm(s[0:len(p_sum)-1],ord=1)

# prediction
def make_prediction(svr_list, s, x, p_sum = [0,10,18,24,28,32]):
    y_pre = numpy.zeros(x.shape[0])
    for i in range(0,len(p_sum)-1):
        clf_regress = svr_list[i]
        y_pre = y_pre + math.fabs(s[i])*clf_regress.predict(x[0:x.shape[0],p_sum[i]:p_sum[i+1]])
    y_pre = y_pre + s[len(p_sum)-1]
    return y_pre

# run finance prediction
def train_and_predict():
    candidate_average = [1,5,10,20,50]
    p = [10,8,6,4,4]
    p_sum = [0,10,18,24,28,32]
    filename = 'sp500_final.csv'

    predict_length = 20
    train_length = 150
    test_length = 20

    trade_date, close_price = read_rawdata_csv(filename)
    trade_date, moving_average = process_rawdata(trade_date,close_price)
    trade_date, return_dict = process_return_list(trade_date,moving_average)
    trade_date, x, y = process_feature(trade_date,return_dict,candidate_average,p)

    daily_turnover = []
    buy_and_hold = []
    cur_index = train_length + test_length
    while cur_index + predict_length < len(trade_date):
        x_train = x[cur_index - train_length - test_length:cur_index - test_length,0:x.shape[1]]
        y_train = y[cur_index - train_length - test_length:cur_index - test_length]
        x_test = x[cur_index - test_length:cur_index,0:x.shape[1]]
        y_test = y[cur_index - test_length:cur_index]
        x_predict = x[cur_index:cur_index + predict_length,0:x.shape[1]]
        y_real = y[cur_index:cur_index + predict_length]

        svr_list, s = train_svr(x_train,y_train,x_test,y_test,p_sum)
        y_predict = make_prediction(svr_list,s,x_predict,p_sum)

        for i in range(0,predict_length):
            buy_and_hold.append(y_real[i])
            if y_real[i]*y_predict[i] > 0:
                daily_turnover.append(math.fabs(y_real[i]))
            else:
                daily_turnover.append(-math.fabs(y_real[i]))
        cur_index = cur_index + predict_length
    accumulated_turnover = numpy.cumsum(daily_turnover)
    accumulated_buy_and_hold = numpy.cumsum(buy_and_hold)
    plt.figure(1)
    plt.plot(accumulated_turnover, label = 'trading by SVR')
    plt.plot(accumulated_buy_and_hold, label = 'trading by buy and hold')
    plt.title('accumulated turnover plot')
    plt.xlabel('trading length')
    plt.ylabel('excessive return')
    plt.legend()
    plt.show()
    
train_and_predict()   