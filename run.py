import argparse
from utils.en_train import EnConfig, EnRun

def main(args):
    EnRun(EnConfig(batch_size=args.batch_size,
                   learning_rate=args.lr,
                   seed=args.seed,
                   fusion_method=args.fusion_method,
                   dataset=args.dataset,
                   num_hidden_layer=args.num_hidden_layers,
                   model_save_path=args.model_save_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed e.g. 1, 10, 42, 100, 123')
    parser.add_argument('--batch_size', type=int, default=8 , help='batch size')
    parser.add_argument('--lr', type=float, default=5e-6, help='learning rate, mosi:5e-6', )
    parser.add_argument('--fusion_method', type=str, default='v2', help='fusion method include v1: concatenation, v2: bidirectional gating network,'
                                                                                     'v3: weighted fusion, v4: transformer fusion')
    parser.add_argument('--dataset', type=str, default='mosi', help='dataset name: mosi')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='the number of hidden layer')
    parser.add_argument('--model_save_path', type=str, default='./saved/', help='path to save model checkpoints')
    args = parser.parse_args()
    main(args)