#!/usr/bin/env python
# -*- coding: utf-8 -*-

with open('FSAKE/train.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace save_checkpoint and add push method
old_save = '''    def save_checkpoint(self, state, is_best):
        torch.save(state, 'asset/checkpoints/{}/'.format(tt.arg.experiment) + 'checkpoint.pth.tar')
        if is_best:
            shutil.copyfile('asset/checkpoints/{}/'.format(tt.arg.experiment) + 'checkpoint.pth.tar',
                            'asset/checkpoints/{}/'.format(tt.arg.experiment) + 'model_best.pth.tar')'''

new_save = '''    def save_checkpoint(self, state, is_best):
        checkpoint_dir = 'asset/checkpoints/{}/'.format(tt.arg.experiment)
        checkpoint_file = checkpoint_dir + 'checkpoint.pth.tar'
        
        torch.save(state, checkpoint_file)
        if is_best:
            shutil.copyfile(checkpoint_file, checkpoint_dir + 'model_best.pth.tar')
    
    def push_checkpoint_to_hf(self):
        """Push current checkpoint to Hugging Face Hub during training."""
        if not HF_AVAILABLE:
            return False
        
        try:
            config_file = '/tmp/fsake_hf_config.json'
            if not os.path.exists(config_file):
                return False
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            HF_REPO_ID = config.get('HF_REPO_ID')
            if not HF_REPO_ID:
                return False
            
            api = HfApi()
            checkpoint_dir = 'asset/checkpoints/{}/'.format(tt.arg.experiment)
            if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
                api.upload_file(
                    path_or_fileobj=checkpoint_dir + 'checkpoint.pth.tar',
                    path_in_repo=f'{tt.arg.experiment}/checkpoint.pth.tar',
                    repo_id=HF_REPO_ID,
                    repo_type='model'
                )
            if os.path.exists(checkpoint_dir + 'model_best.pth.tar'):
                api.upload_file(
                    path_or_fileobj=checkpoint_dir + 'model_best.pth.tar',
                    path_in_repo=f'{tt.arg.experiment}/model_best.pth.tar',
                    repo_id=HF_REPO_ID,
                    repo_type='model'
                )
            return True
        except Exception as e:
            print(f'[INFO] HF push failed: {str(e)[:40]}')
            return False'''

content = content.replace(old_save, new_save)

# Add push call in training loop
old_loop = '''                }, is_best)

            tt.log_step(global_step=self.global_step)'''

new_loop = '''                }, is_best)
                self.push_checkpoint_to_hf()

            tt.log_step(global_step=self.global_step)'''

content = content.replace(old_loop, new_loop)

with open('FSAKE/train.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Successfully added HF push functionality!')
