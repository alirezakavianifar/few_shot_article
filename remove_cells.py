import re

with open('fsake_colab_guide.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the three redundant cells - use exact pattern
content = re.sub(r'<VSCode\.Cell id="#VSC-03d0ee64" language="python">.*?</VSCode\.Cell>', '', content, flags=re.DOTALL)
content = re.sub(r'<VSCode\.Cell id="#VSC-2ef0d7e5" language="markdown">.*?</VSCode\.Cell>', '', content, flags=re.DOTALL)
content = re.sub(r'<VSCode\.Cell id="#VSC-849848b0" language="python">.*?</VSCode\.Cell>', '', content, flags=re.DOTALL)

with open('fsake_colab_guide.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)

print('Cells removed successfully')