import matplotlib.pyplot as plt

directory_string = "CMaps/"
filetype = ".txt"
prefix = "RUL_"
filenames = ["FD001", "FD002", "FD003", "FD004"]

filepaths = [directory_string + prefix + filename + filetype for filename in filenames]

data = []

for filepath in filepaths:
    with open(filepath) as f:
        file_data = f.readlines()
        file_data = [string.replace(" \n", '') for string in file_data]
        file_data = [int(x) for x in file_data]
        data.append(file_data)


fig1, ax1 = plt.subplots()
ax1.set_title('RUL distribution in files')
ax1.set_ylabel('RUL')
ax1.boxplot(data, labels=filenames)
plt.savefig('RUL_destributions.png')