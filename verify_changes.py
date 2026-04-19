#!/usr/bin/env python
# Verification script for FSAKE train.py modifications

with open('FSAKE/train.py', 'r') as f:
    content = f.read()

# Check key elements
checks = {
    'HF_AVAILABLE import': 'HF_AVAILABLE' in content,
    'push_checkpoint_to_hf method': 'push_checkpoint_to_hf' in content,
    'load_checkpoint method': 'def load_checkpoint' in content,
    'iteration tracking': "'iteration': self.global_step" in content,
}

print('=' * 60)
print('FSAKE Train.py Implementation Status')
print('=' * 60)
for key, result in checks.items():
    status = '✅' if result else '❌'
    print(f'{status} {key}: {result}')
print('=' * 60)
