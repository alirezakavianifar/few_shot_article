from torchtools import *
from data import MiniImagenetLoader,TieredImagenetLoader,Cub200Loader,CifarFsLoader
from model import EmbeddingImagenet, Unet,Unet2
import shutil
import os
import random
import time

# ── Hugging Face Hub auto-checkpoint (optional, activated via env vars) ──────
# Set these env vars before running train.py to enable automatic HF pushes:
#   HF_TOKEN      – your HuggingFace write token
#   HF_USERNAME   – your HuggingFace username  (default: derived from token)
#   HF_REPO_NAME  – repo name                  (default: fsake-checkpoints)
# The notebook's Step 5A cell already writes these to /tmp/fsake_hf_config.json
# and the training cell sources them as environment variables.
_HF_ENABLED = False
_hf_api      = None
_HF_REPO_ID  = None

def _init_hf():
    """Initialise Hugging Face Hub if credentials are available."""
    global _HF_ENABLED, _hf_api, _HF_REPO_ID
    try:
        import json
        token    = os.environ.get('HF_TOKEN', '')
        username = os.environ.get('HF_USERNAME', '')
        repo     = os.environ.get('HF_REPO_NAME', 'fsake-checkpoints')

        # Fall back to the JSON config written by the notebook setup cell
        if not token or not username:
            cfg_path = '/tmp/fsake_hf_config.json'
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
                username = username or cfg.get('HF_USERNAME', '')
                repo_id  = cfg.get('HF_REPO_ID', '')
                if repo_id:
                    repo = repo_id.split('/')[-1]

        if not username:
            return  # No credentials – silently skip

        from huggingface_hub import HfApi, login
        if token:
            login(token=token, add_to_git_credential=False)
        _hf_api     = HfApi()
        _HF_REPO_ID = f'{username}/{repo}'
        _HF_ENABLED = True
        print(f'[HF] Checkpoint auto-push ENABLED → {_HF_REPO_ID}')
    except Exception as e:
        print(f'[HF] Auto-push disabled ({e})')


def _push_checkpoint_to_hf(exp_name, filename):
    """Upload a single checkpoint file to the HF repo (non-blocking best-effort)."""
    if not _HF_ENABLED:
        return
    local_path = f'asset/checkpoints/{exp_name}/{filename}'
    if not os.path.exists(local_path):
        return
    try:
        _hf_api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f'{exp_name}/{filename}',
            repo_id=_HF_REPO_ID,
            repo_type='model',
        )
        print(f'[HF] ✓ Pushed {exp_name}/{filename} → {_HF_REPO_ID}')
    except Exception as e:
        print(f'[HF] ✗ Push failed for {filename}: {e}')


def _pull_checkpoint_from_hf(exp_name):
    """Download checkpoint.pth.tar (preferred) and/or model_best.pth.tar from HF."""
    if not _HF_ENABLED:
        return
    from huggingface_hub import hf_hub_download

    local_ckpt = f'asset/checkpoints/{exp_name}/checkpoint.pth.tar'
    local_best = f'asset/checkpoints/{exp_name}/model_best.pth.tar'

    if not os.path.exists(local_ckpt):
        try:
            cached = hf_hub_download(
                repo_id=_HF_REPO_ID,
                filename=f'{exp_name}/checkpoint.pth.tar',
                repo_type='model',
            )
            shutil.copy(cached, local_ckpt)
            print(f'[HF] ✓ Pulled checkpoint.pth.tar from {_HF_REPO_ID} → {local_ckpt}')
        except Exception as e:
            print(f'[HF] No checkpoint.pth.tar on HF ({e})')

    if not os.path.exists(local_best):
        try:
            cached = hf_hub_download(
                repo_id=_HF_REPO_ID,
                filename=f'{exp_name}/model_best.pth.tar',
                repo_type='model',
            )
            shutil.copy(cached, local_best)
            print(f'[HF] ✓ Pulled model_best.pth.tar from {_HF_REPO_ID} → {local_best}')
        except Exception as e:
            print(f'[HF] No model_best.pth.tar on HF ({e})')

class ModelTrainer(object):
    def __init__(self,
                 enc_module,
                 unet_module,
                 data_loader):
        # set encoder and unet
        self.enc_module = enc_module.to(tt.arg.device)#enc_module = EmbeddingImagenet(emb_size=tt.arg.emb_size=128)
        self.unet_module = unet_module.to(tt.arg.device)#unet or unet2


        if tt.arg.num_gpus > 1:  #GPU
            print('Construct multi-gpu model ...')
            self.enc_module = nn.DataParallel(self.enc_module, device_ids=[0, 1], dim=0)
            self.unet_module = nn.DataParallel(self.unet_module, device_ids=[0, 1], dim=0)

            print('done!\n')

        # get data loader
        self.data_loader = data_loader

        # set module parameters
        self.module_params = list(self.enc_module.parameters()) + list(self.unet_module.parameters())

        # set optimizer
        self.optimizer = optim.Adam(params=self.module_params,
                                    lr=tt.arg.lr,
                                    weight_decay=tt.arg.weight_decay)

        # set loss
        self.node_loss = nn.NLLLoss()

        self.global_step = 0
        self.val_acc = 0
        self.test_acc = 0

    def load_checkpoint(self, ckpt_path=None):
        """Load weights, optimizer, and global_step from disk. Prefer checkpoint.pth.tar."""
        exp_dir = 'asset/checkpoints/{}'.format(tt.arg.experiment)
        if ckpt_path is None:
            ckpt_path = os.path.join(exp_dir, 'checkpoint.pth.tar')
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(exp_dir, 'model_best.pth.tar')
        if not os.path.exists(ckpt_path):
            return False

        dev = tt.arg.device
        if isinstance(dev, str):
            map_location = torch.device(dev)
        else:
            map_location = dev

        state = torch.load(ckpt_path, map_location=map_location, weights_only=False)

        def _align_keys(module, saved_sd):
            cur = module.state_dict()
            if not saved_sd or not cur:
                return saved_sd
            sk0 = next(iter(saved_sd.keys()))
            ck0 = next(iter(cur.keys()))
            if sk0 == ck0:
                return saved_sd
            if ck0.startswith('module.') and not sk0.startswith('module.'):
                return {f'module.{k}': v for k, v in saved_sd.items()}
            if sk0.startswith('module.') and not ck0.startswith('module.'):
                return {(k[7:] if k.startswith('module.') else k): v for k, v in saved_sd.items()}
            return saved_sd

        enc_sd = _align_keys(self.enc_module, state['enc_module_state_dict'])
        unet_sd = _align_keys(self.unet_module, state['unet_module_state_dict'])
        self.enc_module.load_state_dict(enc_sd)
        self.unet_module.load_state_dict(unet_sd)

        if 'optimizer' in state and state['optimizer'] is not None:
            try:
                self.optimizer.load_state_dict(state['optimizer'])
            except Exception as e:
                print(f'[RESUME] Warning: could not load optimizer state ({e})')

        self.global_step = int(state.get('iteration', 0))
        self.val_acc = float(state.get('best_val_acc', state.get('val_acc', 0.0)))

        print(f'[RESUME] Loaded {ckpt_path}')
        print(f'[RESUME] iteration={self.global_step}, best_val_acc={self.val_acc:.4f}')
        return True

    def train(self):#
        val_acc = self.val_acc

        # set edge mask (to distinguish support and query edges)
        num_supports = tt.arg.num_ways * tt.arg.num_shots
        num_queries = tt.arg.num_queries
        num_samples = num_supports + num_queries

        time_start=time.time()

        # for each iteration
        for iter in range(self.global_step + 1, tt.arg.train_iteration + 1):
            # init grad
            self.optimizer.zero_grad()

            # set current step
            self.global_step = iter

            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader['train'].get_task_batch(num_tasks=tt.arg.meta_batch_size,
                                                                     num_ways=tt.arg.num_ways,
                                                                     num_shots=tt.arg.num_shots,
                                                                     num_queries=int(tt.arg.num_queries /tt.arg.num_ways),
                                                                     seed=iter + tt.arg.seed)


            # set as single data
            full_data = torch.cat([support_data, query_data], 1)
            full_label = torch.cat([support_label, query_label], 1)
            full_edge = self.label2edge(full_label)

            # set init edge
            init_edge = full_edge.clone()
            init_edge[:, num_supports:, :] = 0.5
            init_edge[:, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, num_supports + i, num_supports + i] = 1.0


            # set as train mode
            self.enc_module.train()
            self.unet_module.train()

            full_data = [self.enc_module(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1)
            one_hot_label = self.one_hot_encode(tt.arg.num_ways, support_label.long())
            query_padding = (1 / tt.arg.num_ways) * torch.ones([full_data.shape[0]] + [num_queries] + [tt.arg.num_ways],
                                                               device=one_hot_label.device)
            one_hot_label = torch.cat([one_hot_label, query_padding], dim=1)
            full_data = torch.cat([full_data, one_hot_label], dim=-1)


            if tt.arg.transductive == True:
                full_node_out5, full_node_out = self.unet_module(init_edge, full_data)
            else:
                # non-transduction
                support_data = full_data[:, :num_supports]
                query_data = full_data[:, num_supports:]
                support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1,
                                                                      1)
                support_data_tiled = support_data_tiled.view(tt.arg.meta_batch_size * num_queries, num_supports,
                                                             -1)
                query_data_reshaped = query_data.contiguous().view(tt.arg.meta_batch_size * num_queries, -1).unsqueeze(1)
                input_node_feat = torch.cat([support_data_tiled, query_data_reshaped], 1)
                input_edge_feat = 0.5 * torch.ones(tt.arg.meta_batch_size, num_supports + 1, num_supports + 1).to(
                    tt.arg.device)  # batch_size x (num_support + 1) x (num_support + 1)

                input_edge_feat[:, :num_supports, :num_supports] = init_edge[:, :num_supports,
                                                                   :num_supports]  # batch_size x (num_support + 1) x (num_support + 1)
                input_edge_feat = input_edge_feat.repeat(num_queries, 1,
                                                         1)  # (batch_size x num_queries) x (num_support + 1) x (num_support + 1)

                # 2. unet
                node_out = self.unet_module(input_edge_feat,
                                            input_node_feat)
                node_out = node_out.view(tt.arg.meta_batch_size, num_queries, num_supports + 1,
                                         tt.arg.num_ways)
                full_node_out = torch.zeros(tt.arg.meta_batch_size, num_samples, tt.arg.num_ways).to(tt.arg.device)
                full_node_out[:, :num_supports, :] = node_out[:, :, :num_supports, :].mean(1)
                full_node_out[:, num_supports:, :] = node_out[:, :, num_supports:, :].squeeze(2)

            # 3. compute loss
            query_node_out5 = full_node_out5[:, -5:]
            node_pred5 = torch.argmax(query_node_out5, dim=-1)
            node_accr5 = torch.sum(torch.eq(node_pred5, full_label[:, num_supports:].long())).float() \
                         / node_pred5.size(0) / num_queries
            node_loss5 = [self.node_loss(data.squeeze(1), label.squeeze(1).long()) for (data, label) in
                          zip(query_node_out5.chunk(query_node_out5.size(1), dim=1),
                              full_label[:, num_supports:].chunk(full_label[:, num_supports:].size(1), dim=1))]
            node_loss5 = torch.stack(node_loss5, dim=0)
            node_loss5 = torch.mean(node_loss5)

            ####################################################################################
            query_node_out = full_node_out[:,num_supports:]
            node_pred = torch.argmax(query_node_out, dim=-1)
            node_accr = torch.sum(torch.eq(node_pred, full_label[:, num_supports:].long())).float() \
                        / node_pred.size(0) / num_queries
            node_loss = [self.node_loss(data.squeeze(1), label.squeeze(1).long()) for (data, label) in
                         zip(query_node_out.chunk(query_node_out.size(1), dim=1), full_label[:, num_supports:].chunk(full_label[:, num_supports:].size(1), dim=1))]
            node_loss = torch.stack(node_loss, dim=0)
            node_loss = torch.mean(node_loss)
            ############################################################################
            node_loss2 = node_loss + 0.5*node_loss5
            #############################################
            node_loss2.backward()

            self.optimizer.step()

            # adjust learning rate
            self.adjust_learning_rate(optimizers=[self.optimizer],
                                      lr=tt.arg.lr,
                                      iter=self.global_step)

            tt.log_scalar('train/loss', node_loss, self.global_step)
            tt.log_scalar('train/node_accr', node_accr, self.global_step)
            tt.log_scalar('train/time', time.time()-time_start, self.global_step)
            ########################################
            tt.log_scalar('train/loss5', node_loss5, self.global_step)
            tt.log_scalar('train/node_accr5', node_accr5, self.global_step)
            tt.log_scalar('train/loss2', node_loss2, self.global_step)
            #######################################

            # evaluation
            if self.global_step % 400 == 0:
                val_acc = self.eval(partition='val')

                is_best = 0

                if val_acc >= self.val_acc:
                    self.val_acc = val_acc
                    is_best = 1

                tt.log_scalar('val/best_accr', self.val_acc, self.global_step)

                self.save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'unet_module_state_dict': self.unet_module.state_dict(),
                    'val_acc': val_acc,
                    'best_val_acc': self.val_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best)

            tt.log_step(global_step=self.global_step)


    def eval(self,partition='test', log_flag=True):
        best_acc = 0

        # set edge mask (to distinguish support and query edges)
        num_supports = tt.arg.num_ways * tt.arg.num_shots
        num_queries = tt.arg.num_queries
        num_samples = num_supports + num_queries

        query_node_accrs = []

        time_start_eval=time.time()

        # for each iteration
        for iter in range(tt.arg.test_iteration // 20):
            # load task data list
            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader[partition].get_task_batch(num_tasks=tt.arg.test_batch_size,
                                                                       num_ways=tt.arg.num_ways,
                                                                       num_shots=tt.arg.num_shots,
                                                                       num_queries=int(tt.arg.num_queries /tt.arg.num_ways),
                                                                       seed=iter)
            '''
            q0 = query_data[:,0,:].clone()
            q1 = query_data[:,1,:].clone()
            query_data[:, 1, :] = q0
            query_data[:, 0, :] = q1
            ql0 = query_label[:,0].clone()
            ql1 = query_label[:,1].clone()
            query_label[:, 1] = ql0
            query_label[:, 0] = ql1
            '''
            # set as single data
            full_data = torch.cat([support_data, query_data], 1)
            full_label = torch.cat([support_label, query_label], 1)
            full_edge = self.label2edge(full_label)



            # set init edge
            init_edge = full_edge.clone()
            init_edge[:, num_supports:, :] = 0.5
            init_edge[:, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, num_supports + i, num_supports + i] = 1.0

            # set as eval mode
            self.enc_module.eval()
            self.unet_module.eval()

            # (1) encode data
            full_data = [self.enc_module(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1)  # batch_size x num_samples x featdim
            one_hot_label = self.one_hot_encode(tt.arg.num_ways, support_label.long())
            query_padding = (1 / tt.arg.num_ways) * torch.ones([full_data.shape[0]] + [num_queries] + [tt.arg.num_ways],
                                                               device=one_hot_label.device)
            one_hot_label = torch.cat([one_hot_label, query_padding], dim=1)
            full_data = torch.cat([full_data, one_hot_label], dim=-1)

            if tt.arg.transductive == True:
                # transduction
                full_node_out5, full_node_out = self.unet_module(init_edge, full_data)
            else:
                # non-transduction
                support_data = full_data[:, :num_supports]  # batch_size x num_support x featdim
                query_data = full_data[:, num_supports:]  # batch_size x num_query x featdim
                support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1,
                                                                      1)  # batch_size x num_queries x num_support x featdim
                support_data_tiled = support_data_tiled.view(tt.arg.test_batch_size * num_queries, num_supports,
                                                             -1)  # (batch_size x num_queries) x num_support x featdim
                query_data_reshaped = query_data.contiguous().view(tt.arg.test_batch_size * num_queries, -1).unsqueeze(
                    1)  # (batch_size x num_queries) x 1 x featdim
                input_node_feat = torch.cat([support_data_tiled, query_data_reshaped],
                                            1)  # (batch_size x num_queries) x (num_support + 1) x featdim

                input_edge_feat = 0.5 * torch.ones(tt.arg.test_batch_size, num_supports + 1, num_supports + 1).to(
                    tt.arg.device)  # batch_size x (num_support + 1) x (num_support + 1)

                input_edge_feat[:, :num_supports, :num_supports] = init_edge[:, :num_supports,
                                                                   :num_supports]  # batch_size x (num_support + 1) x (num_support + 1)
                input_edge_feat = input_edge_feat.repeat(num_queries, 1,
                                                         1)  # (batch_size x num_queries) x (num_support + 1) x (num_support + 1)

                # 2. unet
                node_out = self.unet_module(input_edge_feat,
                                            input_node_feat)
                node_out = node_out.view(tt.arg.test_batch_size, num_queries, num_supports + 1,
                                         tt.arg.num_ways)
                full_node_out = torch.zeros(tt.arg.test_batch_size, num_samples, tt.arg.num_ways).to(tt.arg.device)
                full_node_out[:, :num_supports, :] = node_out[:, :, :num_supports, :].mean(1)
                full_node_out[:, num_supports:, :] = node_out[:, :, num_supports:, :].squeeze(2)

            # 3. compute loss
            query_node_out = full_node_out[:, num_supports:]
            node_pred = torch.argmax(query_node_out, dim=-1)
            node_accr = torch.sum(torch.eq(node_pred, full_label[:, num_supports:].long())).float() \
                        / node_pred.size(0) / num_queries

            query_node_accrs += [node_accr.item()]
            print('time cost',time.time()-time_start_eval,'s')

        # logging
        if log_flag:
            tt.log('---------------------------')
            tt.log_scalar('{}/node_accr'.format(partition), np.array(query_node_accrs).mean(), self.global_step)

            tt.log('evaluation: total_count=%d, accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                   (iter,
                    np.array(query_node_accrs).mean() * 100,
                    np.array(query_node_accrs).std() * 100,
                    1.96 * np.array(query_node_accrs).std() / np.sqrt(
                        float(len(np.array(query_node_accrs)))) * 100))
            tt.log('---------------------------')

        return np.array(query_node_accrs).mean()

    def adjust_learning_rate(self, optimizers, lr, iter):
        new_lr = lr * (0.5 ** (int(iter / tt.arg.dec_lr)))

        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    def label2edge(self, label):
        # get size
        num_samples = label.size(1)
        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)
        edge = torch.eq(label_i, label_j).float().to(tt.arg.device)
        return edge

    def one_hot_encode(self, num_classes, class_idx):
        # Move indices to CPU for indexing torch.eye (which is on CPU)
        class_idx_cpu = class_idx.cpu() if hasattr(class_idx, 'cpu') else class_idx
        # Create one-hot and move to target device
        one_hot = torch.eye(num_classes)[class_idx_cpu].to(tt.arg.device)
        return one_hot

    def save_checkpoint(self, state, is_best):
        ckpt_file = 'asset/checkpoints/{}/checkpoint.pth.tar'.format(tt.arg.experiment)
        torch.save(state, ckpt_file)
        # Always push latest training state so Colab/HF resume is not tied to new bests only
        _push_checkpoint_to_hf(tt.arg.experiment, 'checkpoint.pth.tar')
        if is_best:
            shutil.copyfile(ckpt_file,
                            'asset/checkpoints/{}/model_best.pth.tar'.format(tt.arg.experiment))
            _push_checkpoint_to_hf(tt.arg.experiment, 'model_best.pth.tar')

def set_exp_name():
    exp_name = 'D-{}'.format(tt.arg.dataset)
    exp_name += '_N-{}_K-{}_Q-{}'.format(tt.arg.num_ways, tt.arg.num_shots,tt.arg.num_queries)
    exp_name += '_B-{}_T-{}'.format(tt.arg.meta_batch_size,tt.arg.transductive)
    exp_name += '_P-{}_Un-{}'.format(tt.arg.pool_mode,tt.arg.unet_mode)
    exp_name += '_SEED-{}'.format(tt.arg.seed)

    return exp_name

if __name__ == '__main__':

    tt.arg.device = 'cuda:0' if tt.arg.device is None else tt.arg.device
    tt.arg.dataset_root = 'dataset'
    tt.arg.dataset = 'mini'
    tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
    tt.arg.num_shots = 1
    tt.arg.num_queries = tt.arg.num_ways*1
    tt.arg.num_supports = tt.arg.num_ways*tt.arg.num_shots
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
    unet2_flag = False # the label of using unet2

    # confirm ks
    if tt.arg.num_shots == 1 and tt.arg.transductive == False:

        if tt.arg.pool_mode == 'support':  # 'support': pooling on support   Max score
            tt.arg.ks = [0.6, 0.5]  # 5->3->1
        elif tt.arg.pool_mode == 'kn':  # left close support node   Nearest
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


    tt.arg.train_iteration = 200000 if tt.arg.dataset == 'tiered' else 100000
    tt.arg.test_iteration = 10000
    tt.arg.test_interval = 20
    tt.arg.test_batch_size = 10
    tt.arg.log_step = 100

    tt.arg.lr = 1e-3
    tt.arg.grad_clip = 5
    tt.arg.weight_decay = 1e-6
    tt.arg.dec_lr = 10000 if tt.arg.dataset == 'mini' else 20000
    tt.arg.dropout = 0.1 if tt.arg.dataset == 'mini' else 0.0

    tt.arg.experiment = set_exp_name() if tt.arg.experiment is None else tt.arg.experiment

    print(set_exp_name())

    # set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    tt.arg.log_dir_user = tt.arg.log_dir if tt.arg.log_dir_user is None else tt.arg.log_dir_user
    tt.arg.log_dir = tt.arg.log_dir_user

    if not os.path.exists('asset/checkpoints'):
        os.makedirs('asset/checkpoints')
    if not os.path.exists('asset/checkpoints/' + tt.arg.experiment):
        os.makedirs('asset/checkpoints/' + tt.arg.experiment)

    # Initialise HF auto-push and try to pull the latest best checkpoint
    _init_hf()
    _pull_checkpoint_from_hf(tt.arg.experiment)

    enc_module = EmbeddingImagenet(emb_size=tt.arg.emb_size)

    if tt.arg.transductive == False:
        if unet2_flag == False:
            unet_module = Unet(tt.arg.ks, tt.arg.in_dim, tt.arg.num_ways, 1)
        else:
            unet_module = Unet2(tt.arg.ks_1, tt.arg.ks_2, mode_1, mode_2, tt.arg.in_dim, tt.arg.num_ways, 1)
    else:
        if unet2_flag == False:
            unet_module = Unet(tt.arg.ks, tt.arg.in_dim, tt.arg.num_ways, tt.arg.num_queries)
        else:
            unet_module = Unet2(tt.arg.ks_1, tt.arg.ks_2, mode_1, mode_2, tt.arg.in_dim, tt.arg.num_ways, tt.arg.num_queries)

    if tt.arg.dataset == 'mini':
        train_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='train')
        valid_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='val')
    elif tt.arg.dataset == 'tiered':
        train_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='train')
        valid_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='val')
    elif tt.arg.dataset == 'cub':
        train_loader = Cub200Loader(root=tt.arg.dataset_root, partition='train')
        valid_loader = Cub200Loader(root=tt.arg.dataset_root, partition='val')
    elif tt.arg.dataset == 'cifar':
        train_loader = CifarFsLoader(root=tt.arg.dataset_root, partition='train')
        valid_loader = CifarFsLoader(root=tt.arg.dataset_root, partition='val')
    else:
        print('Unknown dataset!')
        raise NameError('Unknown dataset!!!')

    data_loader = {'train': train_loader,
                   'val': valid_loader
                   }

    # create trainer
    trainer = ModelTrainer(enc_module=enc_module,
                           unet_module=unet_module,
                           data_loader=data_loader)

    trainer.load_checkpoint()

    trainer.train()



