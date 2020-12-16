import time
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision



device = 'cuda' if torch.cuda.is_available() else 'cpu'


def l2_loss(preds, y):
    assert (preds.shape[0] == y.shape[0])
    y_mat = torch.zeros_like(preds)
    y_inds = torch.arange(y.shape[0])
    y_mat[y_inds, y] = 1.
    diff = preds-y_mat
    loss = torch.norm(diff, p=2, dim=1).mean()
    return loss


def l2_predict(preds, return_norms=False):
    gt = torch.eye(preds.shape[1]).to(device)
    p_tile = preds.view(preds.shape[0], 1, -1).repeat(1, preds.shape[1], 1)
    norms = torch.norm((p_tile-gt), p=2, dim=2)
    _, ps = norms.min(1)
    if return_norms:
        return ps, norms
    return ps


def train_l2(
        net,
        epochs,
        batch_size,
        lr,
        reg,
        log_every_n=50,
):
    """Train a deep clustering network
    """
    # Setup
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.875,
                          weight_decay=reg, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1,
            milestones=[int(epochs*0.25), int(epochs*0.5), int(epochs*0.75)])


    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    global_steps = 0
    start = time.time()

    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = l2_loss(outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            predicted = l2_predict(outputs)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_steps += 1

            if global_steps % log_every_n == 0:
                end = time.time()
                num_examples_per_second = log_every_n * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                start = time.time()

        scheduler.step()

        """
        Start the testing code.
        """
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = l2_loss(outputs, targets)

                test_loss += loss.item()
                predicted = l2_predict(outputs)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total
        print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving...")
            torch.save(net.state_dict(), "net_before_pruning.pt")


def test_l2(net, **kwargs):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = l2_loss(outputs, targets)

            test_loss += loss.item()
            predicted = l2_predict(outputs)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    if 'f' in kwargs:
        kwargs["f"].write("Test Loss=%.4f, Test accuracy=%.4f\n" % (test_loss / (num_val_steps), val_acc))
    else:
        print("Test Loss=%.4f, Test accuracy=%.4f" % (test_loss / (num_val_steps), val_acc))


def gather_l2_metrics(net, **kwargs):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=100, shuffle=False, num_workers=16)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])
    testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_targs = 10

    net.eval()
    stats = {'train':{'correct':0, 'loss':0., 'total':0, 'norms':10*[torch.empty((0,))], 'preds':10*[torch.empty((0,))]},
             'test': {'correct':0, 'loss':0., 'total':0, 'norms':10*[torch.empty((0,))], 'preds':10*[torch.empty((0,))]}}
    all_norms = torch.empty((0,))
    with torch.no_grad():
        for loader, name in zip([trainloader, testloader], ['train', 'test']):
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = l2_loss(outputs, targets)

                stats[name]['loss'] += loss.item()
                predicted, norms = l2_predict(outputs, return_norms=True)
                norms = norms.cpu()
                stats[name]['total'] += targets.size(0)
                stats[name]['correct'] += predicted.eq(targets).sum().item()
                for i in range(num_targs):
                    stats[name]['norms'][i] = torch.cat(
                        (stats[name]['norms'][i], norms[targets==i]), dim=0)
                    stats[name]['preds'][i] = torch.cat(
                        (stats[name]['preds'][i], predicted[targets==i].cpu()), dim=0)
            stats[name]['num_steps'] = len(loader)
            stats[name]['accuracy'] = (stats[name]['correct'] /
                                       stats[name]['total'])
    if 'f' in kwargs:
        for name in ['train', 'test']:
            kwargs["f"].write("%s Loss=%.4f, Test accuracy=%.4f\n" % (name,
                stats[name]['loss'] / stats[name]['num_steps'],
                stats[name]['accuracy']))
    else:
        for name in ['train', 'test']:
            print("%s Loss=%.4f, Test accuracy=%.4f\n" % (name,
                stats[name]['loss'] / stats[name]['num_steps'],
                stats[name]['accuracy']))
    return stats
