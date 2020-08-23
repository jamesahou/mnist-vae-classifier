import argparse

parser = argparse.ArgumentParser(description="MNIST VAE CLASSIFIER")
parser.add_argument('--test', default=1, type=int)
args = parser.parse_args()

print(args.test)

args.test = 2

print(args.test)


print(args.test * 2)