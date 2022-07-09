import pandas as pd
from matplotlib import pyplot as plt
import argparse
import os.path as osp


parser = argparse.ArgumentParser(description='Plotting Metrics')

# training
parser.add_argument('--data_dir', default='', type=str,
                    help='Directory of the CSV file to plot')
args = parser.parse_args()
plt.rcParams["figure.figsize"] = [10.00, 5.0]
plt.rcParams["figure.autolayout"] = True
columns = ["Step", "Total Loss", "MSE Loss", "VLB Loss"]
# df = pd.read_csv("input.csv", usecols=columns)
df = pd.read_csv(osp.join(args.data_dir, "progress.csv"))
# df = df[df.step <= 1e5]
# print("Contents in csv file:\n", df)
plt.plot(df.step, df.loss, label=columns[1])
plt.plot(df.step, df.mse, label=columns[2])
plt.legend()
plt.savefig(osp.join(args.data_dir, 'loss.png'), bbox_inches='tight')
plt.show()
plt.plot(df.step, df.vb, label=columns[3])
plt.legend()
plt.savefig(osp.join(args.data_dir, 'loss_vlb.png'), bbox_inches='tight')
plt.show()
