# import subprocess

# out = subprocess.run(['python3', 'testing.py', '-p', '../src5/outfile.txt', '-l', '../training_data/labels.txt'], capture_output=True, text=True)
# out = out.stdout
# out = out.split()[2]

# print(float(out))

l = 56
with open('file.txt', 'w') as f:
    f.write(str(l))
