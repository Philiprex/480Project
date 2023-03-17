import pandas as pd
import preprocessing

def getcc():
    with open("data/BankChurners.csv") as file:
        data = pd.read_csv(file)

    data = data.drop("CLIENTNUM", axis=1)

    labels = data["Dependent_count"]
    data = data.drop("Dependent_count", axis=1)

    train_len = round(len(data), 0)
    x_train, y_train = data[:train_len], labels[:train_len]
    x_test, y_test = data[train_len:], labels[train_len:]

    x_trans = preprocessing.process(x_train)

    return x_trans, y_train


def getFruit():
    with open("data/Date_Fruit_Datasets.csv") as file:
        data = pd.read_csv(file)

    labels = data["Class"]
    data = data.drop("Class", axis=1)

    train_len = round(len(data), 0)
    x_train, y_train = data[:train_len], labels[:train_len]
    x_test, y_test = data[train_len:], labels[train_len:]

    x_trans = preprocessing.process(x_train)

    return x_trans, y_train


def getDrugs():
    with open("data/drug200.csv") as file:
        data = pd.read_csv(file)

    labels = data["Drug"]
    data = data.drop("Drug", axis=1)

    train_len = round(len(data), 0)
    x_train, y_train = data[:train_len], labels[:train_len]
    x_test, y_test = data[train_len:], labels[train_len:]

    x_trans = preprocessing.process(x_train)

    return x_trans, y_train
