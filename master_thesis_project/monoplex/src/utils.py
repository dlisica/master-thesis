import pickle

def load_graph(dataset):
    return load_pickle(path=f'monoplex/data/graphs/{dataset}_graph')

def load_pickle(path):
    with open(path, 'rb') as pickle_file:
        G = pickle.load(pickle_file)
    return G

def save_pickle(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def format_number(number):
    return "{:,}".format(round(number))
