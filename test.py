from BeatDataset import ContrastiveDataset
datapth = "C:/Users/hyukh/Desktop/robot_extract/2022_beat_tracking/datapath/simac"


trdata = ContrastiveDataset(datapth)

print(trdata.data)