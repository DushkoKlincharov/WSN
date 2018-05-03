import csv
import matplotlib.pyplot as plt

def import_data():
    reader = csv.reader(open('PM10-Kavadarci-03.18.csv', mode = 'r'))
    data = list(reader)[2:]
    # timestamps = [x for [x, _] in data]
    measurements = [float(y) for [_, y] in data if y is not '']
    return measurements

def moving_average_helper(lst, n):
    if len(lst) < n:
        return sum(lst) / len(lst)
    return sum(lst[len(lst) - n : ]) / float(n)

def first_order_lp_helper(lst, n=0):
    if len(lst) == 1:
        return lst[0]
    return 2 * lst[len(lst) - 1] - lst[len(lst) - 2]

def second_order_lp_helper(lst, n=0):
    if len(lst) == 1:
        return lst[0]
    elif len(lst) == 2:
        return first_order_lp_helper(lst)
    return 2.5 * lst[len(lst) - 1] - 2 * lst[len(lst) - 2] + 0.5 * lst[len(lst) - 3]

def prediction_algorithm(measurements, thresholds, algorithm, n=0):
    last_sent = measurements[0]
    dbs = []
    sent_measurements = []
    mse = []
    for i in range(len(thresholds)):
        dbs.append([last_sent])
        mse.append(0)
        counter = 1
        for m in measurements[1:]:
            if abs(algorithm(dbs[i], n) - m) > thresholds[i]:
                counter += 1
                last_sent = m
            dbs[i].append(last_sent)
            mse[i] += (abs(last_sent - m) ** 2)
        mse[i] /= len(measurements)
        sent_measurements.append(counter / len(measurements) * 100)
    return [sent_measurements, mse]

def moving_average(measurements, thresholds, n):
    return prediction_algorithm(measurements, thresholds, moving_average_helper, n)

def first_order_lp(measurements, thresholds, n=0):
    return prediction_algorithm(measurements, thresholds, first_order_lp_helper)

def second_order_lp(measurements, thresholds, n=0):
    return prediction_algorithm(measurements, thresholds, second_order_lp_helper)

def simulate(measurements, thresholds):
    for i in range(1, 4):
        m = moving_average(measurements, thresholds, i)[0]
        plt.plot(thresholds, m, label = "MA({})".format(i))
    plt.plot(thresholds, first_order_lp(measurements, thresholds)[0], label = 'first order LP')
    plt.plot(thresholds, second_order_lp(measurements, thresholds)[0], label = 'second order LP')
    plt.xlabel('Threshold')
    plt.ylabel('% of transmissions')
    plt.legend()
    plt.title('Time series data predictors')
    plt.show()
    for i in range(1, 4):
        e = moving_average(measurements, thresholds, i)[1]
        plt.plot(thresholds, e, label = "MA({})".format(i))
    plt.plot(thresholds, first_order_lp(measurements, thresholds)[1], label = 'first order LP')
    plt.plot(thresholds, second_order_lp(measurements, thresholds)[1], label = 'second order LP')
    plt.xlabel('Threshold')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Time series data predictors')
    plt.show()
   
if __name__ == '__main__':
    measurements = import_data()
    thresholds = [5, 10, 15, 20, 25, 30]
    simulate(measurements, thresholds)