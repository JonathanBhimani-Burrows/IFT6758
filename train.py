import argparse
import utils.dataloader
from models.relations_agglomerator import relations_agglomerator
from utils.save_model import save_model

def train():
    _, _, relation_data, profile_data, _ = utils.dataloader.load_data(args.i, train=True)

    agglo = relations_agglomerator()

    agglo.train(relation_data, profile_data)

    save_model("relation_agglo", args.o, agglo)
    print("Model saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', help="path to the input data", type=str,
                        default='/home/mila/teaching/user06/Train/')
    parser.add_argument('-o', help="path to the output data", type=str,
                        default='/home/mila/teaching/user06/submissions/IFT6758/models/')
    args = parser.parse_args()

    args.i = "data\\Train\\"
    args.o = "results"

    print('input path:', args.i)
    print('output path:', args.o)
    train()
