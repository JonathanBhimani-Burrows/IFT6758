import os
import pickle

def save_model(name, savedir, model):
    name = name + ".mdl"
    savepath = os.path.join(savedir, name)

    pickle_out = open(savepath, "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()


def load_model(savepath):
    pickle_in = open(savepath, "rb")
    model = pickle.load(pickle_in)
    pickle_in.close()

    return model
