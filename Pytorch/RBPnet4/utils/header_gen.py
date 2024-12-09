from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
import os

from quantization import InputEncoder
from RailwaySet import RailwaySet

#########################
# supported model candidates

candidates = [
                'binput-pg',
             ]
#########################

# Argument parser.
parser = argparse.ArgumentParser(description='Encode CIFAR-10 Test Images')
parser.add_argument('--model_id', '-id', type=int, default=0)
parser.add_argument('--batch_size', '-b', type=int, default=10)
parser.add_argument('--data_dir', '-d', type=str, default='../data',
                    help='path to the dataset directory')
parser.add_argument('--which_gpus', '-gpu', type=str, default='0', help='which gpus to use')
parser.add_argument('--output_dir', '-o', type=str, default='./encoded_images',
                    help='path to save the encoded images')
args = parser.parse_args()
_ARCH = candidates[args.model_id]

# Load the CIFAR-10 dataset.
def load_cifar10():
    normalize = transforms.Normalize([0.4213, 0.3929, 0.3594], [0.2761, 0.2633, 0.2506])
    transform_test_list = [
        transforms.ToTensor()]

    if 'binput' not in _ARCH:
        transform_test_list.append(normalize)

    transform_test = transforms.Compose(transform_test_list)

    testset = RailwaySet(root=args.data_dir, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2,
                                             pin_memory=True, drop_last=False)

    return testloader

def save_intermediate(x, labels, output_dir='encoder_input_image', batch_index=0):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the intermediate data directly without normalization
    x1 = x.cpu().detach().numpy()
    file_path = os.path.join(output_dir, f'image_hw_all_host_new_{batch_index}.bin')
    x1.tofile(file_path)

    # Save labels to a text file
    labels_path = os.path.join(output_dir, 'labels.txt')
    with open(labels_path, 'a') as f:
        for label in labels.cpu().numpy():
            f.write(f"{label}\n")

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Available GPUs: {}".format(torch.cuda.device_count()))

    # Prepare Data
    print("Loading the data.")
    testloader = load_cifar10()

    # Load the pre-trained ResNet50 model with updated weights parameter
    teacher = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    for param in teacher.parameters():
        param.requires_grad = False

    # img_mean and img_std for denormalization
    img_mean = torch.tensor([0.4213, 0.3929, 0.3594]).view(1, -1, 1, 1).to(device)
    img_std = torch.tensor([0.2761, 0.2633, 0.2506]).view(1, -1, 1, 1).to(device)

    # Create InputEncoder instance
    input_size = (args.batch_size, 3, 224, 224)
    encoder = InputEncoder(input_size=input_size, resolution=8).to(device)

    # Process only the first 10 images
    for i, data in enumerate(testloader, 0):
        if i >= 1:  # Since each batch contains 10 images
            break

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # Pass through teacher network
        with torch.no_grad():
            lesson = teacher(inputs)

        # Denormalize the input
        inputs = inputs * img_std + img_mean

        # Pass through the encoder
        x = encoder(inputs)

        # Save the intermediate encoded data
        save_intermediate(x, labels, output_dir=args.output_dir, batch_index=i)

if __name__ == "__main__":
    main()
