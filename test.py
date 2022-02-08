from BeatDataset import ContrastiveDataset
datapth = "./datapath/simac"


trdata = ContrastiveDataset(datapth)


print(trdata[0][0].shape)