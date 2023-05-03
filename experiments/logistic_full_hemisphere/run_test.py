from test import test

for subject in range(1, 9):
    for side in ["left", "right"]:
        test(f"subj0{subject}", side)