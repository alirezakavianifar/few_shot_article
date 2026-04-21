from torchtools import *
from data import MiniImagenetLoader,TieredImagenetLoader,Cub200Loader,CifarFsLoader
from model import EmbeddingImagenet, Unet,Unet2, LatentMediatorUnet
import shutil
import os
import random
from train import ModelTrainer



if __name__ == '__main__':

    tt.arg.device = 'cuda:3' if tt.arg.device is None else tt.arg.device
    tt.arg.dataset_root = 'dataset'
    tt.arg.dataset = 'mini'
    tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
    tt.arg.num_shots = 1
    tt.arg.num_queries = tt.arg.num_ways * 1
    tt.arg.num_supports = tt.arg.num_ways * tt.arg.num_shots
    tt.arg.transductive = True if tt.arg.transductive is None else tt.arg.transductive
    if tt.arg.transductive == False:
        tt.arg.meta_batch_size = 20
    else:
        tt.arg.meta_batch_size = 40
    tt.arg.seed = 222 if tt.arg.seed is None else tt.arg.seed
    tt.arg.num_gpus = 1

    # model parameter related
    tt.arg.emb_size = 128
    tt.arg.in_dim = tt.arg.emb_size + tt.arg.num_ways

    tt.arg.pool_mode = 'support'
    tt.arg.unet_mode = 'addold' if tt.arg.unet_mode is None else tt.arg.unet_mode # 'addold'/'noold'
    tt.arg.interaction_block = 'baseline' if tt.arg.interaction_block is None else str(tt.arg.interaction_block).lower()
    tt.arg.mediator_tokens = 8 if tt.arg.mediator_tokens is None else int(tt.arg.mediator_tokens)
    tt.arg.mediator_layers = 2 if tt.arg.mediator_layers is None else int(tt.arg.mediator_layers)
    tt.arg.mediator_heads = 4 if tt.arg.mediator_heads is None else int(tt.arg.mediator_heads)
    tt.arg.mediator_dropout = 0.1 if tt.arg.mediator_dropout is None else float(tt.arg.mediator_dropout)
    unet2_flag = False  # the label of using unet2

    # confirm ks
    if tt.arg.num_shots == 1 and tt.arg.transductive == False:
        if tt.arg.pool_mode == 'support':  # 'support': pooling on support
            tt.arg.ks = [0.6, 0.5]  # 5->3->1
        elif tt.arg.pool_mode == 'kn':  # left close support node
            tt.arg.ks = [0.6, 0.5]  # 5->3->1
        else:
            print('wrong mode setting!!!')
            raise NameError('wrong mode setting!!!')
    elif tt.arg.num_shots == 5 and tt.arg.transductive == False:
        if tt.arg.pool_mode == 'way':  # 'way' pooling on support by  way
            tt.arg.ks_1 = [0.6, 0.5]  # 5->3->1
            mode_1 = 'way'
            tt.arg.ks_2 = [0.6, 0.5]  # 5->3->1 # supplementary pooling for fair comparing
            mode_2 = 'support'
            unet2_flag = True
        elif tt.arg.pool_mode == 'kn':
            tt.arg.ks_1 = [0.6, 0.5]  # 5->3->1
            mode_1 = 'way&kn'
            tt.arg.ks_2 = [0.6, 0.5]  # 5->3->1 # supplementary pooling for fair comparing
            mode_2 = 'kn'
            unet2_flag = True
        else:
            print('wrong mode setting!!!')
            raise NameError('wrong mode setting!!!')

    elif tt.arg.num_shots == 1 and tt.arg.transductive == True:
        if tt.arg.pool_mode == 'support':  # 'support': pooling on support
            tt.arg.ks = [0.6, 0.5]  # 5->3->1
        elif tt.arg.pool_mode == 'kn':  # left close support node
            tt.arg.ks = [0.6, 0.5]  # 5->3->1
        else:
            print('wrong mode setting!!!')
            raise NameError('wrong mode setting!!!')

    elif tt.arg.num_shots == 5 and tt.arg.transductive == True:
        if tt.arg.pool_mode == 'way':  # 'way' pooling on support by  way
            tt.arg.ks_1 = [0.6, 0.5]  # 5->3->1
            mode_1 = 'way'
            tt.arg.ks_2 = [0.6, 0.5]  # 5->3->1 # supplementary pooling for fair comparing
            mode_2 = 'support'
            unet2_flag = True
        elif tt.arg.pool_mode == 'kn':
            tt.arg.ks_1 = [0.2]  # 5->1
            mode_1 = 'way&kn'
            tt.arg.ks_2 = [0.2]  # 5->1 # supplementary pooling for fair comparing
            mode_2 = 'kn'
            unet2_flag = True
        else:
            print('wrong mode setting!!!')
            raise NameError('wrong mode setting!!!')

    else:
        print('wrong shot and T settings!!!')
        raise NameError('wrong shot and T settings!!!')

    # train, test parameters
    tt.arg.train_iteration = 200000 if tt.arg.dataset == 'tiered' else 100000
    tt.arg.test_iteration = 10000
    tt.arg.test_interval = 5000
    tt.arg.test_batch_size = 10
    tt.arg.log_step = 1000

    tt.arg.lr = 1e-3
    tt.arg.grad_clip = 5
    tt.arg.weight_decay = 1e-6
    tt.arg.dec_lr = 10000 if tt.arg.dataset == 'mini' else 20000
    tt.arg.dropout = 0.1 if tt.arg.dataset == 'mini' else 0.0

    # set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tt.arg.exp_name = 'D-{}'.format(tt.arg.dataset)
    tt.arg.exp_name += '_N-{}_K-{}_Q-{}'.format(tt.arg.num_ways, tt.arg.num_shots, tt.arg.num_queries)
    tt.arg.exp_name += '_B-{}_T-{}'.format(tt.arg.meta_batch_size, tt.arg.transductive)
    tt.arg.exp_name += '_P-{}_Un-{}'.format(tt.arg.pool_mode, tt.arg.unet_mode)
    if tt.arg.interaction_block == 'lmt':
        tt.arg.exp_name += '_IB-lmt_M{}L{}H{}'.format(
            tt.arg.mediator_tokens,
            tt.arg.mediator_layers,
            tt.arg.mediator_heads,
        )
    tt.arg.exp_name += '_SEED-{}'.format(tt.arg.seed)

    print(tt.arg.exp_name)

    enc_module = EmbeddingImagenet(emb_size=tt.arg.emb_size)

    if tt.arg.interaction_block == 'lmt':
        unet_module = LatentMediatorUnet(
            in_dim=tt.arg.in_dim,
            num_classes=tt.arg.num_ways,
            num_queries=tt.arg.num_queries,
            mediator_tokens=tt.arg.mediator_tokens,
            mediator_layers=tt.arg.mediator_layers,
            mediator_heads=tt.arg.mediator_heads,
            mediator_dropout=tt.arg.mediator_dropout,
        )
    else:
        if tt.arg.transductive == False:
            if unet2_flag == False:
                unet_module = Unet(tt.arg.ks, tt.arg.in_dim, tt.arg.num_ways, 1)
            else:
                unet_module = Unet2(tt.arg.ks_1, tt.arg.ks_2, mode_1, mode_2, tt.arg.in_dim, tt.arg.num_ways, 1)
        else:
            if unet2_flag == False:
                unet_module = Unet(tt.arg.ks, tt.arg.in_dim, tt.arg.num_ways, tt.arg.num_queries)
            else:
                unet_module = Unet2(tt.arg.ks_1, tt.arg.ks_2, mode_1, mode_2, tt.arg.in_dim, tt.arg.num_ways,
                                    tt.arg.num_queries)


    if tt.arg.dataset == 'mini':
        test_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='test')
    elif tt.arg.dataset == 'tiered':
        test_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='test')
    elif tt.arg.dataset == 'cub':
        test_loader = Cub200Loader(root=tt.arg.dataset_root, partition='test')
    elif tt.arg.dataset == 'cifar':
        test_loader = CifarFsLoader(root=tt.arg.dataset_root, partition='test')
    else:
        print('Unknown dataset!')
        raise NameError('Unknown dataset!!!')


    data_loader = {'test': test_loader}

    # create trainer
    tester = ModelTrainer(enc_module=enc_module,
                          unet_module=unet_module,
                          data_loader=data_loader)

    # PyTorch>=2.6 defaults torch.load(..., weights_only=True), which breaks
    # legacy checkpoints containing optimizer/metadata pickled objects.
    # These checkpoints are produced locally by train.py in this repo.
    checkpoint = torch.load(
        'asset/checkpoints/{}/'.format(tt.arg.exp_name) + 'model_best.pth.tar',
        map_location=tt.arg.device,
        weights_only=False,
    )


    tester.enc_module.load_state_dict(checkpoint['enc_module_state_dict'])
    print("load pre-trained enc_nn done!")

    # initialize gnn pre-trained
    tester.unet_module.load_state_dict(checkpoint['unet_module_state_dict'])
    print("load pre-trained unet done!")

    tester.val_acc = checkpoint['val_acc']
    tester.global_step = checkpoint['iteration']

    print(tester.global_step,tester.val_acc)

    tester.eval(partition='test')
