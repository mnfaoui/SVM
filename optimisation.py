# Importation de la bilbiotheque Cifra10

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
file = "./cifar-10-batches-py/data_batch_1"
print(unpickle(file))