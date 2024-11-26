filename = "best_para_2024-11-24_18-50-39"
with open(f"{filename}.txt", 'r') as f1:
    with open(f"{filename}_truncated.txt", 'w') as f2:
        for line1 in f1:
            if "#" in line1:
                line2 = "#\n"
            else:
                line2 = float(line1.strip())
                # line2 = f"{line2:.4e}\n"
                line2 = round(line2, 3)
                line2 = str(line2) + '\n'
            f2.write(line2)