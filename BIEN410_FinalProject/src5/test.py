import subprocess

out = subprocess.run(['python3', '../testing/testing.py', '-p', 'outfile.txt', '-l', '../training_data/labels.txt'], capture_output=True, text=True)
out = out.stdout
out = out.split()[2]

print(float(out))
