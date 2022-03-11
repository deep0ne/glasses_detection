import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

from eyeglasses import Predictor


def args_parsing():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_data_path', required=True, help='input data path')

    parser.add_argument('-n', '--n_images', type=int,
                        help='number of images to predict',
                        default=10000)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    model = torch.load('mobilev3_model.pt', map_location='cpu')
    model.eval()

    args = args_parsing()
    test = Predictor(args.input_data_path, model, args.n_images)
    test.get_labels()