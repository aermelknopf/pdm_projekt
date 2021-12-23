



dir_string = 'CMaps/'
filenames = ['FD001', 'FD002', 'FD003', 'FD004']
filetype = '.txt'
prefixes = ['train_', 'test_', 'RUL_']

data = []

for prefix in prefixes:
    id_offset = 0
    prefix_data = []

    for filename in filenames:
        current_file = dir_string + prefix + filename + filetype

        with open(current_file) as f:
            file_data = f.readlines()
            if prefix == 'RUL_':
                file_data = [string.replace("  \n", '') for string in file_data]  # RUL files end with only ' \n'
            else:
                file_data = [string.replace("  \n", '') for string in file_data]    # remove '  \n' at end of files

            file_data = [string.split(" ") for string in file_data]
            file_data = [[float(x) for x in element if x != '\n'] for element in file_data]

            if prefix != 'RUL_':
                for inner_list in file_data:
                    inner_list[0] = int(inner_list[0]) + id_offset
                    inner_list[1] = int(inner_list[1])

                id_offset = file_data[-1][0]

        prefix_data += file_data

    with open(dir_string + prefix + 'all' + filetype, 'w') as out:
        out_data = [" ".join(map(str, line)) for line in prefix_data]

        for index, line in enumerate(out_data):
            if index != len(out_data) - 1:
                out.write(line + '\n')
            else:
                out.write(line)


    # data += prefix_data



b = 42