from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from utils import util as util
import utils.quantization as q
from utils.RailwaySet import RailwaySet

import numpy as np
import os, sys, time
import warnings
import argparse
import copy

# ignore "corrupt EXIF data" warnings in the console
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


#########################
# supported model candidates

candidates = [
                'binput-pg-quant-shortcut',
             ]
#########################


#----------------------------
# Argument parser.
#----------------------------
parser = argparse.ArgumentParser(description='PyTorch RailwaySet Training')
parser.add_argument('--model_id', '-id', type=int, default=0)
parser.add_argument('--gtarget', '-g', type=float, default=0.0)
parser.add_argument('--init_lr', '-lr', type=float, default=5e-4)
parser.add_argument('--batch_size', '-b', type=int, default=91)
parser.add_argument('--num_epoch', '-e', type=int, default=20)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5)
parser.add_argument('--last_epoch', '-last', type=int, default=-1)
parser.add_argument('--finetune', '-f', action='store_true', help='finetune the model')
parser.add_argument('--save', '-s', action='store_true', help='save the model')
parser.add_argument('--test', '-t', action='store_true', help='test only')
parser.add_argument('--resume', '-r', type=str, default=None,
                    help='path of the model for resuming training')
parser.add_argument('--load_states', '-l', type=str, default=None,
                    help='path of states to the optimizer and scheduler')
parser.add_argument('--data_dir', '-d', type=str,
                    default='./data',
                    help='path to the dataset directory')
parser.add_argument('--which_gpus', '-gpu', type=str, default='0', help='which gpus to use')

args = parser.parse_args()
_ARCH = candidates[args.model_id]
drop_last = True if 'binput' in _ARCH else False


#----------------------------
# Load the RailwaySet dataset.
#----------------------------

def load_dataset():
    normalize = transforms.Normalize(mean=[0.421, 0.393, 0.359], std=[0.276, 0.263, 0.251])
    crop_scale = 0.08
    lighting_param = 0.1

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        util.Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Load the training dataset
    train_dataset = RailwaySet(root=args.data_dir, train=True, transform=train_transforms)

    # Load the validation dataset
    val_dataset = RailwaySet(root=args.data_dir, train=False, transform=val_transforms)

    # Create the data loaders
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=drop_last
    )

    valloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=drop_last
    )

    print(f"Number of batches in trainloader: {len(trainloader)}")
    print(f"Number of batches in valloader: {len(valloader)}")

    return trainloader, valloader


#----------------------------
# Define the model.
#----------------------------

def generate_model(model_arch):
    if 'binput-pg-quant-shortcut' in model_arch:
        import fracbnn_imagenet as m
        return m.ReActNet(
                batch_size=args.batch_size,
                num_gpus=torch.cuda.device_count()
               )
    else:
        raise NotImplementedError("Model architecture is not supported.")


#----------------------------
# Train the network.
#----------------------------

def train_model(trainloader, valloader, net, optimizer, scheduler, start_epoch, num_epoch, device):
    # Define the loss function
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    best_acc = 0.
    best_model = copy.deepcopy(net.state_dict())  # Initialize with the starting state
    states = {'epoch': start_epoch,
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler.state_dict()}

    for epoch in range(start_epoch, num_epoch):

        # Set printing functions
        batch_time = util.AverageMeter('Time/batch', ':.2f')
        losses = util.AverageMeter('Loss', ':6.2f')
        top1 = util.AverageMeter('Acc@1', ':6.2f')
        progress = util.ProgressMeter(
            len(trainloader),
            [losses, top1, batch_time],
            prefix="Epoch: [{}]".format(epoch + 1)
        )

        # Switch the model to training mode
        net.train()

        print('Current learning rate = {}'.format(optimizer.param_groups[0]['lr']))

        end = time.time()
        for i, data in enumerate(trainloader, 0):

            # Get the inputs; data is a tuple of (inputs, labels)
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs, _ = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Measure accuracy and record loss
            acc1, acc2 = accuracy(outputs, labels, topk=(1, 2))

            # Convert tensors to Python scalars before updating meters
            losses.update(loss.item(), labels.size(0))
            top1.update(acc1.item(), labels.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 99:
                # Print statistics every 100 mini-batches each epoch
                progress.display(i)

        # Update the learning rate every epoch
        scheduler.step()

        # Evaluate the model on the validation set
        epoch_acc = test_accu(valloader, net, device)
        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(net.state_dict())  # Save the best model state
            states = {'epoch': epoch + 1,
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict()}  # Save training states

            # Save the model and training states if required
            if args.save:
                print("Saving the trained model.")
                this_file_path = os.path.dirname(os.path.abspath(__file__))
                save_folder = os.path.join(this_file_path, 'save_ImageNet_model')
                util.save_models(best_model, save_folder, suffix=_ARCH + '-finetune' if args.finetune else _ARCH)
                util.save_states(states, save_folder, suffix=_ARCH + '-finetune' if args.finetune else _ARCH)

        print("Best test accuracy so far: {:.1f}".format(best_acc))

    print('Finished Training')


def binary_classification_metrics(outputs, labels):
    '''
    Computes True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
    for binary classification.
    '''
    class_0_count = (labels == 0).sum().item()
    class_1_count = (labels == 1).sum().item()

    print(f"Class 0 count: {class_0_count}, Class 1 count: {class_1_count}")

    with torch.no_grad():
        # Assuming the model outputs raw logits with shape [batch_size, 2]
        # Convert to probabilities for the positive class (class 1)
        probs = torch.sigmoid(outputs[:, 1])  # Take the probability of the positive class (class 1)

        # Binary predictions (0 or 1) by rounding the probabilities
        preds = torch.round(probs)

        # Convert to CPU tensors for easier handling
        preds = preds.cpu()
        labels = labels.cpu()

        # Metrics for class 1 (Positive class)
        TP = ((preds == 1) & (labels == 1)).sum().item()  # True Positives
        FP = ((preds == 1) & (labels == 0)).sum().item()  # False Positives
        FN = ((preds == 0) & (labels == 1)).sum().item()  # False Negatives

        # Metrics for class 0 (Negative class)
        TN = ((preds == 0) & (labels == 0)).sum().item()  # True Negatives
        FN_0 = ((preds == 1) & (
                    labels == 0)).sum().item()  # False Negatives (for class 0, prediction is 1 but true is 0)
        FP_0 = ((preds == 0) & (
                    labels == 1)).sum().item()  # False Positives (for class 0, prediction is 0 but true is 1)

        return {
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'FN_0': FN_0, 'FP_0': FP_0
        }


#----------------------------
# Test accuracy.
#----------------------------

def accuracy(outputs, labels, topk=(1,)):
    '''
    Computes the accuracy over the k top predictions for
    the specified values of k
    '''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Using reshape instead of view
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def test_accu(valloader, net, device):
    top1 = util.AverageMeter('Acc@1', ':6.2f')
    net.eval()

    print(f"test Number of batches in valloader: {len(valloader)}")

    total_TP, total_FP, total_FN = 0, 0, 0  # Initialize counters for TP, FP, and FN for class 1
    total_TN, total_FP_0, total_FN_0 = 0, 0, 0  # Initialize counters for TN, FP, FN for class 0

    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            images, labels = data[0].to(device), data[1].to(device)

            outputs, _ = net(images)

            # Compute accuracy
            acc1 = accuracy(outputs, labels, topk=(1,))[0]  # top-1 accuracy
            top1.update(acc1.item(), images.size(0))

            # Compute TP, FP, FN, TN
            metrics = binary_classification_metrics(outputs, labels)
            total_TP += metrics['TP']
            total_FP += metrics['FP']
            total_FN += metrics['FN']

            total_TN += metrics['TN']
            total_FP_0 += metrics['FP_0']
            total_FN_0 += metrics['FN_0']

            print(f"Class 1 (Positive): TP = {metrics['TP']}, FP = {metrics['FP']}, FN = {metrics['FN']}")
            print(f"Class 0 (Negative): TN = {metrics['TN']}, FP_0 = {metrics['FP_0']}, FN_0 = {metrics['FN_0']}")

        # Compute precision, recall, and F1-score for class 1 (positive class)
        precision_1 = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
        recall_1 = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
        f1_score_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if precision_1 + recall_1 > 0 else 0

        # Compute precision, recall, and F1-score for class 0 (negative class)
        precision_0 = total_TN / (total_TN + total_FN_0) if total_TN + total_FN_0 > 0 else 0
        recall_0 = total_TN / (total_TN + total_FP_0) if total_TN + total_FP_0 > 0 else 0
        f1_score_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if precision_0 + recall_0 > 0 else 0

        # Print metrics
        print(f"Class 1 (Positive) Precision: {precision_1:.4f}, Recall: {recall_1:.4f}, F1-Score: {f1_score_1:.4f}")
        print(f"Class 0 (Negative) Precision: {precision_0:.4f}, Recall: {recall_0:.4f}, F1-Score: {f1_score_0:.4f}")

    print(f" * Acc@1 {top1.avg:.1f}")
    return top1.avg




#----------------------------
# Report sparsity in PG
#----------------------------

def sparsity(valloader, net, device):
    num_out, num_high = [], []

    def _report_sparsity(m):
        classname = m.__class__.__name__
        if isinstance(m, q.PGBinaryConv2d):
            num_out.append(m.num_out.item())
            num_high.append(m.num_high.item())

    net.eval()
    net.apply(_report_sparsity)
    cnt_out = np.zeros(len(num_out))
    cnt_high = np.zeros(len(num_high))
    num_out, num_high = [], []

    batch_cnt = 50
    with torch.no_grad():
        start = time.time()
        for data in valloader:
            batch_cnt -= 1
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            net.apply(_report_sparsity)
            cnt_out += np.array(num_out)
            cnt_high += np.array(num_high)
            num_out = []
            num_high = []
            if batch_cnt == 0:
                break
        elapsed_time = time.time() - start
    if np.sum(cnt_out) > 0:
        sparsity_percentage = 100.0 - (np.sum(cnt_high) * 100.0 / np.sum(cnt_out))
    else:
        sparsity_percentage = float('nan')
    print(f'Sparsity of the update phase: {sparsity_percentage:.1f} %  Elapsed time = {elapsed_time:.1f} sec')


#----------------------------
# Remove the saved placeholder
#----------------------------

def remove_placeholder(state_dict):
    from collections import OrderedDict
    temp_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if 'encoder.placeholder' in key:
            pass
        elif 'teacher' in key:
            pass
        else:
            temp_state_dict[key] = value
    return temp_state_dict

#####

def benchmark_inference(valloader, net, device, num_runs=10):
    net.eval()  # Set the network to evaluation mode
    total_time = 0.0
    with torch.no_grad():  # Disable gradient computation for inference
        for i, data in enumerate(valloader):
            images, labels = data[0].to(device), data[1].to(device)

            # Measure the time taken for the inference
            for _ in range(num_runs):
                start_time = time.time()
                outputs, _ = net(images)
                end_time = time.time()
                total_time += (end_time - start_time)

            # Only run the benchmark on the first batch
            break

    avg_time_per_inference = total_time / (num_runs * images.size(0))  # Calculate average time per image
    print(f"Average inference time per image (over {num_runs} runs): {avg_time_per_inference:.6f} seconds")



#----------------------------
# Main function.
#----------------------------

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Available GPUs: {}".format(torch.cuda.device_count()))

    print("Create {} model.".format(_ARCH))
    net = generate_model(_ARCH)
    if torch.cuda.device_count() > 1:
        print("Activate multi GPU support.")
        net = nn.DataParallel(net)
    net.to(device)

    #------------------
    # Load model params
    #------------------
    if args.resume is not None:
        model_path = args.resume
        if os.path.exists(model_path):
            print("@ Load trained model from {}.".format(model_path))
            state_dict = torch.load(model_path, map_location=device)
            state_dict = remove_placeholder(state_dict)
            net.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Model not found.")

    #-----------------
    # Prepare Data
    #-----------------
    print("Loading the data.")
    trainloader, valloader = load_dataset()

    #-----------------
    # Test
    #-----------------
    if args.test:
        print("Mode: Test only.")
        test_accu(valloader, net, device)
        if 'pg' in _ARCH:
            sparsity(valloader, net, device)

        # Benchmark GPU inference time on a single image
        benchmark_inference(valloader, net, device)



    #-----------------
    # Finetune
    #-----------------
    elif args.finetune:
        print("num epochs = {}".format(args.num_epoch))
        initial_lr = args.init_lr
        print("init lr = {}".format(initial_lr))
        optimizer = optim.Adam(net.parameters(),
                          lr = initial_lr,
                          weight_decay=0.)

        print("Use linear learning rate decay.")
        lambda1 = lambda epoch : (1.0-epoch/args.num_epoch)
        scheduler = optim.lr_scheduler.LambdaLR(
                            optimizer,
                            lr_lambda=lambda1,
                            last_epoch=args.last_epoch)

        start_epoch=0
        if args.load_states is not None:
            states_path = args.load_states
            if os.path.exists(states_path):
                print("@ Load training states from {}.".format(states_path))
                states = torch.load(states_path)
                start_epoch = states['epoch']
                optimizer.load_state_dict(states['optimizer'])
                scheduler.load_state_dict(states['scheduler'])
            else:
                raise ValueError("Saved states not found.")

        print("Start finetuning.")
        train_model(trainloader, valloader, net,
                    optimizer, scheduler, start_epoch,
                    args.num_epoch, device)
        _ = test_accu(valloader, net, device)

    #-----------------
    # Train
    #-----------------
    else:
        print("num epochs = {}".format(args.num_epoch))
        initial_lr = args.init_lr
        print("init lr = {}".format(initial_lr))
        optimizer = optim.Adam(net.parameters(),
                          lr = initial_lr,
                          weight_decay=args.weight_decay)

        print("Use linear learning rate decay.")
        lambda1 = lambda epoch : (1.0-epoch/args.num_epoch)
        scheduler = optim.lr_scheduler.LambdaLR(
                            optimizer,
                            lr_lambda=lambda1,
                            last_epoch=args.last_epoch)

        start_epoch = 0
        if args.load_states is not None:
            states_path = args.load_states
            if os.path.exists(states_path):
                print("@ Load training states from {}.".format(states_path))
                states = torch.load(states_path)
                start_epoch = states['epoch']
                optimizer.load_state_dict(states['optimizer'])
                scheduler.load_state_dict(states['scheduler'])
            else:
                raise ValueError("Saved states not found.")

        print("Start training.")
        train_model(trainloader, valloader, net,
                    optimizer, scheduler, start_epoch,
                    args.num_epoch, device)
        _ = test_accu(valloader, net, device)


if __name__ == "__main__":
    main()
