from train import train

for subject in range(1,9):
    for hemisphere in ["left", "right"]:
        train(subject, hemisphere)