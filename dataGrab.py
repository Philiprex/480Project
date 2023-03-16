import pandas as pd
import preprocessing

def getcc():
    with open("data/BankChurners.csv") as file:
        data = pd.read_csv(file)

    data = data.drop("CLIENTNUM", axis=1)

    labels = data["Dependent_count"]
    data = data.drop("Dependent_count", axis=1)

    data = preprocessing.process(data)

    return data, labels


def getFruit():
    with open("data/Date_Fruit_Datasets.csv") as file:
        data = pd.read_csv(file)

    labels = data["Class"]
    data = data.drop("Class", axis=1)

    data = preprocessing.process(data)

    return data, labels


def getDrugs():
    with open("data/drug200.csv") as file:
        data = pd.read_csv(file)

    labels = data["Drug"]
    data = data.drop("Drug", axis=1)

    data = preprocessing.process(data)

    return data, labels
