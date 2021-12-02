import argparse
import os
import math
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from fvcore.nn import FlopCountAnalysis

from lib.data import get_dataset_ft
from lib.utils import accuracy, AverageMeter

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class PrunedBasicBlock(nn.Module):
    def __init__(self, inplanes, midplanes, outplanes, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, midplanes)
        self.bn1 = norm_layer(midplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(midplanes, outplanes)
        self.bn2 = norm_layer(outplanes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon = 0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def get_flops_and_params(model):
    model = model.eval()
    with torch.no_grad():
        # flop count
        inp = torch.zeros(1, 3, 224, 224)
        flops = FlopCountAnalysis(model.cpu(), inp).total()

        # param count
        params = sum(p.numel() for p in model.parameters())
        return flops, params


def residual_objective(mat):
    return 0.5 * torch.sum(mat**2)


def prune_basic_block(input_data, og_block, ratio=0.5, num_iter=20, bs=128, verbose=False):
    if verbose:
        print(f'\nPruning Settings: ratio {ratio}, iters {num_iter}, data ex {input_data.shape[0]}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    og_block = og_block.to(device)

    # compute size of pruned block and create it
    in_channels = og_block.conv1.in_channels
    out_channels = og_block.conv1.out_channels
    assert in_channels == out_channels
    pruned_channels = int(og_block.conv1.out_channels * ratio)
    pruned_block = PrunedBasicBlock(in_channels, pruned_channels, out_channels)
    pruned_block = pruned_block.to(device)

    # compute dense hidden representation
    og_hidden = []
    for i in range(0, input_data.shape[0], bs):
        mb_input = input_data[i: i + bs, :]
        mb_input = mb_input.to(device)
        with torch.no_grad():
            mb_hid = og_block.conv1(mb_input)
            mb_hid = og_block.bn1(mb_hid)
            mb_hid = og_block.relu(mb_hid)
            og_hidden.append(mb_hid.cpu())
    with torch.no_grad():
        og_hidden = torch.cat(og_hidden, dim=0)
        agg_hidden = torch.sum(og_hidden, dim=(0, 2, 3))

    # main pruning loop
    pruned_indices = set([])
    pruned_indexer = None
    for t in range(num_iter):
        # compute importance with automatic differentiation (chunked into mini-batches)
        importance = None
        for i in range(0, og_hidden.shape[0], bs):
            mb_hid = og_hidden[i:i + bs, :]
            mb_hid = mb_hid.to(device)
            mb_hid.requires_grad = True # track gradient on layer input to determine importance
            og_output = og_block.conv2(mb_hid)
            og_output = og_block.bn2(og_output)
            if len(pruned_indices) > 0:
                with torch.no_grad():
                    pruned_hid = mb_hid[:, pruned_indexer, :, :]
                    pruned_output = pruned_block.conv2(pruned_hid)
                    pruned_output = pruned_block.bn2(pruned_output)
                residual = residual_objective(og_output - pruned_output.detach())
                residual.backward()
            else:
                residual = residual_objective(og_output)
                residual.backward()

            tmp_imp = mb_hid.grad.detach().cpu()
            with torch.no_grad():
                tmp_imp = torch.sum(tmp_imp, dim=0)
                if importance is None:
                    importance = tmp_imp
                else:
                    importance += tmp_imp
                
            mb_hid.grad = None
            og_block.conv2.zero_grad()
            og_block.bn2.zero_grad()

        # find most important neurons, merge with previous active set, then threshold
        with torch.no_grad():
            importance = torch.sum(importance, dim=(1, 2))
            imp_idxs = torch.argsort(importance, descending=True)[:2*pruned_channels]
            tmp_imp_channels = set(imp_idxs.cpu().tolist())
            bigger_set = tmp_imp_channels.union(pruned_indices)
            indexer = torch.LongTensor(sorted(list(bigger_set)))
            hidden_sizes = agg_hidden[indexer]
            new_pruned_indices = torch.argsort(hidden_sizes, descending=True)[:pruned_channels]
            new_pruned_indices = set(indexer[new_pruned_indices].cpu().tolist())
            pruned_indices = new_pruned_indices
            pruned_indexer = torch.LongTensor(sorted(list(pruned_indices))).to(device)
            pruned_block.conv1.weight.data = og_block.conv1.weight.data[pruned_indexer, :]
            pruned_block.bn1.weight.data = og_block.bn1.weight.data[pruned_indexer]
            pruned_block.bn1.bias.data = og_block.bn1.bias.data[pruned_indexer]
            pruned_block.conv2.weight.data = og_block.conv2.weight.data[:, pruned_indexer, :]
            pruned_block.bn2.weight.data = og_block.bn2.weight.data
            pruned_block.bn2.bias.data = og_block.bn2.bias.data 
    return pruned_block


def run_ft(args, model, epochs, lr, criterion=None, optimizer=None, use_lr_sched=False, verbose=False):
    if verbose:
        print(f'\n\nRunning FT for {epochs} epoch(s), lr {lr}, sched {use_lr_sched}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    ft_load, test_load, n_class = get_dataset_ft('imagenet', args.batch_size,
            args.workers, args.data_path)
    
    if criterion is None:
        criterion = CrossEntropyLabelSmooth(1000).cuda()
    
    if optimizer is None:
        no_wd_params, wd_params = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if ".bn" in name or '.bias' in name:
                    no_wd_params.append(param)
                else:
                    wd_params.append(param)
        no_wd_params = nn.ParameterList(no_wd_params)
        wd_params = nn.ParameterList(wd_params)
        optimizer = torch.optim.SGD([
                                    {'params': no_wd_params, 'weight_decay':0.},
                                    {'params': wd_params, 'weight_decay': args.wd},
                                ], lr, momentum=args.momentum, nesterov=True)
    
    tloss, tprec1 = validate(test_load, model, criterion, verbose=True)
    if verbose:
        print(f'Epoch 0 Test Loss/Acc: {tloss:.2f}/{tprec1:.2f}')

    test_accs = [tprec1]
    test_losses = [tloss]
    trn_losses = []
    trn_accs = []
    for e in range(epochs):
        if verbose:
            print(f'Running FT Epoch {e+1}/{epochs}')

        # use cosine learning rate schedule
        if use_lr_sched:
            new_lr = 0.5 * lr * (1 + math.cos(math.pi * e / epochs))
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr
            if verbose:
                print(f'\n\nChanging LR to: {new_lr}\n\n')

        losses = AverageMeter()
        accs = AverageMeter()
        model = model.to(device)
        model.train()
        for i, (inputs, targets) in enumerate(ft_load): 
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
            accs.update(prec1.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))
        trn_losses.append(losses.avg)
        trn_accs.append(accs.avg)
        tloss, tprec1 = validate(test_load, model, criterion)
        if verbose:
            print(f'\n\nEpoch {e+1} Test Loss/Acc: {tloss:.2f}/{tprec1:.2f}\n')
        test_accs.append(tprec1)
        test_losses.append(tloss)

    metrics = {
        'trn_accs': trn_accs,
        'trn_losses': trn_losses,
        'test_accs': test_accs,
        'test_losses': test_losses,
    }
    return model, metrics


def validate(val_loader, model, criterion, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
    if verbose:
        pruneflops, pruneparams = get_flops_and_params(model)
        print("Prune Flops, Prune Params")
        print(pruneflops, pruneparams)
        print(f'Test Acc.: {top1.avg:.4f}')
    return losses.avg, top1.avg
     

def prune_rn34_imagenet():
    parser = argparse.ArgumentParser(description='i-SpaSP prune for rn34')
    parser.add_argument('--save-dir', type=str, default='./results')
    parser.add_argument('--exp-name', type=str, default='ispasp_rn34_prune_00')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--data-path', type=str, default='/data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-cs-batches', type=int, default=5)
    parser.add_argument('--num-cs-iter', type=int, default=20)
    parser.add_argument('--layer1-ratio', type=float, default=0.4)
    parser.add_argument('--layer2-ratio', type=float, default=0.4)
    parser.add_argument('--layer3-ratio', type=float, default=0.4)
    parser.add_argument('--layer4-ratio', type=float, default=1.0)
    parser.add_argument('--block-ft-epochs', type=int, default=1)
    parser.add_argument('--block-ft-lr', type=float, default=1e-2)
    parser.add_argument('--prune-ft-lr', type=float, default=1e-2)
    parser.add_argument('--prune-ft-epochs', type=int, default=90)
    parser.add_argument('--use-lr-sched', action='store_true', default=False)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--prune-last-layer', action='store_true', default=False)
    parser.add_argument('--pruned-path', type=str, default=None)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.pruned_path is None:
        # prune each layer of ResNet34 with i-SpaSP based on pre-defined ratios
        # for each group of convolutional blocks within the network
        prune_perf_mets = {
            'layer1': [],
            'layer2': [],
            'layer3': [],
            'layer4': [],
        }
        prune_load, _, _ = get_dataset_ft('imagenet', args.batch_size,
                args.workers, args.data_path)
    
        pruned_model = torchvision.models.resnet34(pretrained=True)
        pruned_model = pruned_model.to(device)    

        # prune layer1
        if args.layer1_ratio < 1.0:
            if args.prune_last_layer:
                max_layers = len(pruned_model.layer1)
            else:
                max_layers = len(pruned_model.layer1) - 1 
            for i in range(1, max_layers):
                with torch.no_grad():
                    pruned_model.eval()
                    full_prune_data = []
                    data_iter = iter(prune_load)
                    for b in range(args.num_cs_batches):
                        data_in = next(data_iter)[0].to(device)

                        tmp_prune_data = pruned_model.maxpool(pruned_model.relu(pruned_model.bn1(pruned_model.conv1(data_in))))
                        for lidx in range(i):
                            tmp_prune_data = pruned_model.layer1[lidx](tmp_prune_data) 
                        full_prune_data.append(tmp_prune_data.cpu())
                    full_prune_data = torch.cat(full_prune_data, dim=0)

                assert pruned_model.layer1[i].downsample is None
                pruned_model.layer1[i] = prune_basic_block(full_prune_data, pruned_model.layer1[i],
                        args.layer1_ratio, args.num_cs_iter, args.batch_size,
                        verbose=args.verbose).to(device)
                if args.block_ft_epochs > 0:
                    pruned_model, tmp_met = run_ft(
                            args, pruned_model, args.block_ft_epochs, args.block_ft_lr,
                            use_lr_sched=False, verbose=args.verbose)
                    prune_perf_mets['layer1'].append(tmp_met)
        else:
            print('No pruning on layer1')
         
        # prune layer2
        if args.layer2_ratio < 1.0:
            if args.prune_last_layer:
                max_layers = len(pruned_model.layer2)
            else:
                max_layers = len(pruned_model.layer2) - 1
            for i in range(1, max_layers):
                with torch.no_grad():
                    pruned_model.eval()
                    full_prune_data = []
                    data_iter = iter(prune_load)
                    for b in range(args.num_cs_batches):
                        data_in = next(data_iter)[0].to(device)

                        # everything before layer being pruned
                        tmp_prune_data = pruned_model.layer1(
                                pruned_model.maxpool(pruned_model.relu(pruned_model.bn1(
                                pruned_model.conv1(data_in)))))
                        for lidx in range(i):
                            tmp_prune_data = pruned_model.layer2[lidx](tmp_prune_data) 
                        full_prune_data.append(tmp_prune_data.cpu())
                    full_prune_data = torch.cat(full_prune_data, dim=0)

                # prune a residual block
                assert pruned_model.layer2[i].downsample is None
                pruned_model.layer2[i] = prune_basic_block(full_prune_data, pruned_model.layer2[i],
                        args.layer2_ratio, args.num_cs_iter, args.batch_size,
                        verbose=args.verbose).to(device)
                if args.block_ft_epochs > 0:
                    pruned_model, tmp_met = run_ft(
                            args, pruned_model, args.block_ft_epochs, args.block_ft_lr,
                            use_lr_sched=False, verbose=args.verbose)
                    prune_perf_mets['layer2'].append(tmp_met)
        else:
            print('No pruning on layer2')
    
        # prune layer3
        if args.layer3_ratio < 1.0:
            if args.prune_last_layer:
                max_layers = len(pruned_model.layer3)
            else:
                max_layers = len(pruned_model.layer3) - 1
            for i in range(1, max_layers):
                with torch.no_grad():
                    pruned_model.eval()
                    full_prune_data = []
                    data_iter = iter(prune_load)
                    for b in range(args.num_cs_batches):
                        data_in = next(data_iter)[0].to(device)

                        # everything before layer being pruned
                        tmp_prune_data = pruned_model.layer2(pruned_model.layer1(
                                pruned_model.maxpool(pruned_model.relu(pruned_model.bn1(
                                pruned_model.conv1(data_in))))))
                        for lidx in range(i):
                            tmp_prune_data = pruned_model.layer3[lidx](tmp_prune_data) 
                        full_prune_data.append(tmp_prune_data.cpu())
                    full_prune_data = torch.cat(full_prune_data, dim=0)

                # prune a residual block
                assert pruned_model.layer3[i].downsample is None
                pruned_model.layer3[i] = prune_basic_block(full_prune_data, pruned_model.layer3[i],
                        args.layer3_ratio, args.num_cs_iter, args.batch_size,
                        verbose=args.verbose).to(device) 
                if args.block_ft_epochs > 0:
                    pruned_model, tmp_met = run_ft(
                            args, pruned_model, args.block_ft_epochs, args.block_ft_lr,
                            use_lr_sched=False, verbose=args.verbose)
                    prune_perf_mets['layer3'].append(tmp_met)
        else:
            print('No pruning on layer3')
    
        # prune layer4
        if args.layer4_ratio < 1.0:
            if args.prune_last_layer:
                max_layers = len(pruned_model.layer4)
            else:
                max_layers = len(pruned_model.layer4) - 1
            for i in range(1, max_layers):
                with torch.no_grad():
                    pruned_model.eval()
                    full_prune_data = []
                    data_iter = iter(prune_load)
                    for b in range(args.num_cs_batches):
                        data_in = next(data_iter)[0].to(device)

                        # everything before layer being pruned
                        tmp_prune_data = pruned_model.layer3(pruned_model.layer2(
                                pruned_model.layer1(pruned_model.maxpool(pruned_model.relu(
                                pruned_model.bn1(pruned_model.conv1(data_in)))))))
                        for lidx in range(i):
                            tmp_prune_data = pruned_model.layer4[lidx](tmp_prune_data) 
                        full_prune_data.append(tmp_prune_data.cpu())
                    full_prune_data = torch.cat(full_prune_data, dim=0)

                # prune a residual block
                assert pruned_model.layer4[i].downsample is None
                pruned_model.layer4[i] = prune_basic_block(full_prune_data, pruned_model.layer4[i],
                        args.layer4_ratio, args.num_cs_iter, args.batch_size,
                        verbose=args.verbose).to(device) 
                if args.block_ft_epochs > 0:
                    pruned_model, tmp_met = run_ft(
                            args, pruned_model, args.block_ft_epochs, args.block_ft_lr,
                            use_lr_sched=False, verbose=args.verbose)
                    prune_perf_mets['layer4'].append(tmp_met)
        else:
            print('No pruning on layer4')
    
        pruneflops, pruneparams = get_flops_and_params(pruned_model)
        
        pre_ft_results = {
            'model': pruned_model.cpu(),
            'prune_perf_mets': prune_perf_mets,
            'args': args,
        }
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(pre_ft_results, os.path.join(args.save_dir, f'{args.exp_name}_no_ft.pth'))
    else:
        # load pruned model from a checkpoint
        pre_ft_results = torch.load(args.pruned_path)
        pruned_model = pre_ft_results['model']
        prune_perf_mets = pre_ft_results['prune_perf_mets']

    if args.prune_ft_epochs > 0:
        criterion = CrossEntropyLabelSmooth(1000).cuda()
        no_wd_params, wd_params = [], []
        for name, param in pruned_model.named_parameters():
            if param.requires_grad:
                if ".bn" in name or '.bias' in name:
                    no_wd_params.append(param)
                else:
                    wd_params.append(param)
        no_wd_params = nn.ParameterList(no_wd_params)
        wd_params = nn.ParameterList(wd_params)
        optimizer = torch.optim.SGD([
	        {'params': no_wd_params, 'weight_decay':0.},
		    {'params': wd_params, 'weight_decay': args.wd},
        ], args.prune_ft_lr, momentum=args.momentum, nesterov=True)

        # run final fine tuning and find metrics
        pruned_model, metrics = run_ft(args, pruned_model, args.prune_ft_epochs, args.prune_ft_lr,
            criterion=criterion, optimizer=optimizer, use_lr_sched=args.use_lr_sched, verbose=args.verbose)
        pruneflops, pruneparams = get_flops_and_params(pruned_model)
        prune_data = {
            'prune_flops': pruneflops,
            'prune_params': pruneparams,
        }

        # save the results
        all_results = {
            'model': pruned_model.cpu(),
            'perf_mets': metrics,
            'prune_mets': prune_data,
            'args': args,
        }
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(all_results, os.path.join(args.save_dir, f'{args.exp_name}.pth'))


if __name__=='__main__':
    prune_rn34_imagenet()
