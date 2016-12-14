import matplotlib.pyplot as plt


def get_losses(file_path):
    print file_path
    file = open(file_path, 'r')
    return file.read().splitlines()


def save_figure(x, y, file_name):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')

    plt.savefig(file_name)


def main():
    # experiments = [
    #     'extended-training',
    #     'extended-training-v2',
    #     'fine-tuning',
    #     'fine-tuning-v2',
    #     'large-network',
    #     'large-network-with-dropout',
    #     'medium-network',
    #     'small-network',
    # ]

    # for experiment in experiments:
    #     losses = get_losses('Experiments/%s/loss-history.txt' % experiment)
    #     epochs = range(len(losses))
    #     save_figure(epochs, losses, '%s.png' % experiment)

    first_losses = get_losses('Experiments/large-network-with-dropout/loss-history.txt')
    second_losses = get_losses('Experiments/extended-training-v2/loss-history.txt')
    losses = first_losses + second_losses
    print len(losses)

    epochs = range(6802)
    save_figure(epochs, losses, '%s.png' % 'extended-training-v2-extension')
        

if __name__ == '__main__':
    main()

