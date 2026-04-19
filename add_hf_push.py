import re

with open('FSAKE/train.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace the save_checkpoint method
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
            
            # Push checkpoint.pth.tar (always - for resuming)
            checkpoint_file = checkpoint_dir + 'checkpoint.pth.tar'
            if os.path.exists(checkpoint_file):
                api.upload_file(
                    path_or_fileobj=checkpoint_file,
                    path_in_repo=f'{tt.arg.experiment}/checkpoint.pth.tar',
                    repo_id=HF_REPO_ID,
                    repo_type='model'
                )
            
            # Push model_best.pth.tar (best model found so far)
            best_file = checkpoint_dir + 'model_best.pth.tar'
            if os.path.exists(best_file):
                api.upload_file(
                    path_or_fileobj=best_file,
                    path_in_repo=f'{tt.arg.experiment}/model_best.pth.tar',
                    repo_id=HF_REPO_ID,
                    repo_type='model'
                )
            
            return True
        except Exception as e:
            print(f'[INFO] HF push failed at iteration {self.global_step}: {str(e)[:40]}')
            return False'''

content = content.replace(old_save, new_save)

# 2. Add the push call after save_checkpoint in training loop
old_train_loop = '''                self.save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'unet_module_state_dict': self.unet_module.state_dict(),
                    'val_acc': val_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best)

            tt.log_step(global_step=self.global_step)'''

new_train_loop = '''                self.save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'unet_module_state_dict': self.unet_module.state_dict(),
                    'val_acc': val_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best)
                
                # Push checkpoint to Hugging Face Hub
                self.push_checkpoint_to_hf()

            tt.log_step(global_step=self.global_step)'''

content = content.replace(old_train_loop, new_train_loop)

with open('FSAKE/train.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('All changes added successfully!')
