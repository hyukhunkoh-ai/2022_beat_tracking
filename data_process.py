from glob import glob
import os

paths = list(glob(os.path.join('./datapath/ballroom', 'label', '*.beats')))

try:
    os.mkdir('./datapath/ballroom/new_label')
except:
    print("already exists")

data_paths = []

for path in paths:
    new_path = path.replace("/label/", "/new_label/")
    new_data = path.replace(".beats", ".wav").replace("/label/", "/data/")
    temp = []
    with open(path, 'r') as fp:
        for index, line in enumerate(fp.readlines()):
            time_in_seconds, beat_number = line.strip('\n').replace('\t', ' ').split(' ')
            time_in_seconds = float(time_in_seconds)
            if index == 0:
                first_time = time_in_seconds

            temp.append([time_in_seconds - 0.005, time_in_seconds + 0.005, 1 if beat_number == "1" else 0])

    last_time = time_in_seconds

    if last_time - first_time >= 12.8 or first_time < 2:
        data_paths.append(new_data)
        with open(new_path, 'w') as fp2:
            for entry in temp:
                fp2.write(str(entry[0]) + "\t" + str(entry[1]) + "\t" + str(entry[2]) + "\n")
    else:
        print("filtered" + new_data)

with open('./datapath/ballroom/new_data.txt', 'w') as fp:
    for data_path in data_paths:
        fp.write(data_path + "\n")
