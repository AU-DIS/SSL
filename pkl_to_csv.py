import pickle
import numpy as np

def open_pkl(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

inses = ['', '_CONE']
for ins in inses:
    import csv
    graph_names  = ['ant', 'football', 'highschool', 'malaria', 'powerlaw_200_50_50', 'renyi_200_50', 'barabasi_200_50']
    for graph_name in graph_names:
        for num in range(1,11):
            name = f'./pkl_results{ins}/{graph_name}/{graph_name}_{num}.pkl'
            res_dict = open_pkl(name)
            data = res_dict[graph_name]
            for size, size_data in data.items():
                with open(f'./csv_results/{graph_name}_{size}_{num}{ins}.csv', 'w') as csv_file:  
                    print(size_data)
                    writer = csv.writer(csv_file, delimiter=" ")
                    writer.writerow(["Perc","Cond", "Acc","BalAcc"])
                    counter = 0
                    for cond, accs in size_data.items():
                        writer.writerow([counter, cond, accs[0], accs[1]])  
                        counter += 0.1


inses = ['', '_CONE']
for ins in inses:         
    import csv
    graph_names  = ['ant', 'football', 'highschool', 'malaria', 'powerlaw_200_50_50', 'renyi_200_50', 'barabasi_200_50']
    for graph_name in graph_names:
        avg_con = {}
        avg_acc = {}
        avg_balacc = {}
        for num in range(1,11):
            name = f'./pkl_results{ins}/{graph_name}/{graph_name}_{num}.pkl'
            res_dict = open_pkl(name)
            data = res_dict[graph_name]
            for size, size_data in data.items():
                if size not in avg_con:
                    avg_con[size] = np.array([0.0 for _ in range(11)])
                    avg_acc[size] = np.array([0.0 for _ in range(11)])
                    avg_balacc[size] = np.array([0.0 for _ in range(11)])
                counter = 0
                for cond, bal_acc in size_data.items():
                    avg_con[size][counter] += cond
                    if bal_acc[0] == 0:avg_acc[size][counter] += 1
                    else: avg_acc[size][counter] += bal_acc[0]
                    if bal_acc[1] == 0: avg_balacc[size][counter] += 1
                    else: avg_balacc[size][counter] += bal_acc[1]
                    counter += 1
                    print(f'counter = {counter}')
                    print(cond, bal_acc)
                print()
        for size in avg_con:
            avg_con[size] = avg_con[size]/10.0
            avg_acc[size] = avg_acc[size]/10.0
            avg_balacc[size] = avg_balacc[size]/10.0
        print(avg_con)
        print(avg_acc)
        print(avg_balacc)

        for size in avg_con:
            with open(f'./csv_results/{graph_name}_{size}_avg{ins}.csv', 'w') as csv_file:  
                writer = csv.writer(csv_file, delimiter=" ")
                writer.writerow(["Perc","Cond", "Acc","BalAcc"])
                counter = 0
                for i in range(len(avg_con[size])):
                    writer.writerow([counter, avg_con[size][i], avg_acc[size][i],  avg_balacc[size][i]]) 
                    counter += 0.1
            
        