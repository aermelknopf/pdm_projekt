import os




if __name__ == "__main__":
    dir = "results2"

    for filename in os.listdir(dir):
        with open(f"{dir}/{filename}", 'r+') as file:
            corrected = [line.replace(',', '') for line in file.readlines()]
            file.writelines(corrected)
