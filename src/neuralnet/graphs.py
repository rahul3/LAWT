import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from common import get_logger

logger = get_logger(__name__)

def training_val_loss(training_examples, val_losses, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(training_examples, val_losses, marker='o')

    plt.title('Validation Loss vs. Training Examples')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Validation Loss')

    plt.xscale('linear')
    plt.yscale('log')

    # Format x-axis
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.xticks(training_examples, rotation=45)

    # Format y-axis
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0e'))

    # Set y-axis ticks
    y_ticks = np.logspace(np.log10(min(val_losses)), np.log10(max(val_losses)), num=5)
    plt.yticks(y_ticks)

    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        path = os.path.join(os.getcwd(), 'training_val_loss.png')
        plt.savefig(path)
            