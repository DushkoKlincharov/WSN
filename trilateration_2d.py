import random as rnd
import math
import matplotlib.pyplot as plt
import numpy as np

def generate_network(N, L, a, interactive):
    nodes_x = [rnd.uniform(0, L) for _ in range(N)]
    nodes_y = [rnd.uniform(0, L) for _ in range(N)]

    nodes = list(zip(nodes_x, nodes_y))
    anchors = nodes[: int(a * N / 100)]
    unknowns = nodes[int(a * N / 100) :]

    if interactive:
        plt.plot([x for (x, y) in anchors], [y for (x, y) in anchors], 'ro', markersize = 3, label = 'anchors')
        plt.plot([x for (x, y) in unknowns], [y for (x, y) in unknowns], 'bo', markersize = 3, label = 'unknowns')
        plt.xlim([0, L])
        plt.ylim([0, L])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=True)
        plt.show()
        print("Anchors: {}, Unknown: {}, Total error: 0, Mean error: 0".format(len(anchors), len(unknowns)))

    return [anchors, unknowns]

def distance(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

def locate(p1, p2, p3, r1, r2, r3): #using trilateration
    A = -2*p1[0] + 2*p2[0]
    B = -2*p1[1] + 2*p2[1]    
    C = r1**2 - r2**2 - p1[0]**2 + p2[0]**2 - p1[1]**2 + p2[1]**2
    D = -2*p2[0] + 2*p3[0]
    E = -2*p2[1] + 2*p3[1]    
    F = r2**2 - r3**2 - p2[0]**2 + p3[0]**2 - p2[1]**2 + p3[1]**2

    x = (C*E - F*B) / (E*A - B*D)
    y = (C*D - A*F) / (B*D - A*E)

    return (x, y)

def anchors_in_range_of_unknown(anchors, unknown, radius):
    result = []
    for a in anchors:
        d = distance(a, unknown)
        if d <= radius:
            result.append((d, a))
    result.sort() # sorted by distance
    return [y for (x, y) in result]

def trilaterate(anchors, unknowns, radius, error, draw_lines_between, total_error, interactive, L):
    discovered_unknowns = []
    discovered_locations = []
    
    for node in unknowns:
        l = anchors_in_range_of_unknown(anchors, node, radius)
        if len(l) >= 3:
            discovered_unknowns.append(node)
            d1 = distance(l[0], node)
            d1 = np.random.uniform(d1 - d1 * error, d1 + d1 * error)
            d2 = distance(l[1], node)
            d2 = np.random.uniform(d2 - d2 * error, d2 + d2 * error)
            d3 = distance(l[2], node)
            d3 = np.random.uniform(d3 - d3 * error, d3 + d3 * error)
            location = locate(l[0], l[1], l[2], d1, d2, d3)

            discovered_locations.append(location)
            draw_lines_between.append((node, location))
            total_error += distance(node, location)

    unknowns = list(set(unknowns) - set(discovered_unknowns))

    if interactive:
        plt.plot([x for (x, y) in anchors], [y for (x, y) in anchors], 'ro', markersize = 3, label = 'anchors')
        plt.plot([x for (x, y) in unknowns], [y for (x, y) in unknowns], 'bo', markersize = 3, label = 'unknowns')
        plt.plot([x for (x, y) in discovered_locations], [y for (x, y) in discovered_locations], 'go', markersize = 3, label = 'discovered')
        for line in draw_lines_between:
            plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], linewidth = 1)
            plt.plot([line[0][0]], line[0][1], 'bo', markersize = 3)
        plt.xlim([0, L])
        plt.ylim([0, L])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=True)
        plt.show()

    anchors = list(set(anchors) | set(discovered_locations))

    if total_error != 0 and interactive:
        print("Anchors: {}, Unknown: {}, Total error: {:.2f}, Mean error: {:.2f}%".format(len(anchors), len(unknowns), total_error, (total_error / len(draw_lines_between) * 100 / radius)))
    if(len(discovered_unknowns) > 0):
        return trilaterate(anchors, unknowns, radius, error, draw_lines_between, total_error, interactive, L)
    else:
        if interactive:
            for line in draw_lines_between:
                plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], linewidth = 1)
                plt.plot([line[0][0]], line[0][1], 'ko', markersize = 3)
                plt.plot([line[1][0]], line[1][1], 'ko', markersize = 3)
            plt.plot([x for (x, y) in anchors], [y for (x, y) in anchors], 'ko', markersize = 3)
            plt.plot([x for (x, y) in unknowns], [y for (x, y) in unknowns], 'bo', markersize = 3)
            plt.xlim([0, L])
            plt.ylim([0, L])
            plt.show()
        if len(draw_lines_between) == 0:
            return 100
        return (total_error / len(draw_lines_between) * 100 / radius)

def loc_error_to_range_error_plot():
    N = int(input("Број на јазли: "))
    L = int(input("Големина на област: "))
    a = int(input("Процент на anchor јазли од вкупниот број јазли (0% - 100%): "))
    range_error = [0, 5, 10, 15, 20, 25, 30]
    radio_range = [20, 30, 40]
    [anchors, unknowns] = generate_network(N, L, a, False)
    for R in radio_range:
        y = []
        for i in range(0, len(range_error)):
            y.append(0)
            for j in range(0, 50):
                y[i] += trilaterate(anchors, unknowns, R*L/100, range_error[i] / 100, [], 0, False, L)
            y[i] /= 50
        plt.plot(range_error, y, label = "R: {}%".format(R))
    plt.xlabel("Шум во сигнал %")
    plt.ylabel("Средна грешка како процент од Radio Range")
    plt.legend()
    plt.title("Број на јазли:{}\nГолемина на област:{}\nПроцент на anchor јазли:{}%".format(N, L, a))
    plt.show()

def loc_error_to_anchors_percentage_plot():
    N = int(input("Број на јазли: "))
    L = int(input("Големина на област: "))
    r = int(input("Шум во сигналот (0% - 100%): "))
    anchor_percentage = [20, 25, 30, 35, 40, 45, 50, 55, 60]
    radio_range = [20, 30, 40]
    for R in radio_range:
        y = []
        for i in range(0, len(anchor_percentage)):
            y.append(0)
            for j in range(0, 100):
                [anchors, unknowns] = generate_network(N, L, anchor_percentage[i], False)
                # print(R, anchor_percentage[i])
                y[i] += trilaterate(anchors, unknowns, R*L/100, r/100, [], 0, False, L)
            y[i] /= 100
        plt.plot(anchor_percentage, y, label = "R: {}%".format(R))
    plt.xlabel("Почетни anchor јазли %")
    plt.ylabel("Средна грешка како процент од Radio Range")
    plt.title("Број на јазли: {}\nГолемина на област: {}\nШум на сигналот: {}%".format(N, L, r))
    plt.legend()
    plt.show()

def simulate():
    N = int(input("Број на јазли: "))
    L = int(input("Големина на област: "))
    a = int(input("Процент на anchor јазли од вкупниот број јазли (0% - 100%): "))
    R = int(input("Радио опсег како процент од големина на областа: "))
    r = int(input("Шум во сигналот (0% - 100%): "))
    [anchors, unknowns] = generate_network(N, L, a, True)
    print("Средна грешка како процент од Range Error: {:.2f}%".format(trilaterate(anchors, unknowns, R*L/100, r/100, [], 0, True, L)))

if __name__ == "__main__":
    # simulate()
    # loc_error_to_anchors_percentage_plot()
    # loc_error_to_range_error_plot()


