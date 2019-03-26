import sys
from pathlib import Path
import pandas as pd
import numpy as np


def cv_results(cv_dir):

	cv_dir = Path(cv_dir)
	split_dirs = sorted([x for x in cv_dir.iterdir() if x.is_dir()])
	
	accuracy = []
	for split in split_dirs:
		
		try:
			val_results = pd.read_csv(split / 'val_results.csv', index_col=0)
		except FileNotFoundError:
			continue
		
		best_epoch = val_results['accuracy'].max()
		if best_epoch < 0.1:
			print(f"Ignoring outlier '{split}'")
			continue
		accuracy.append(best_epoch)
	
	return accuracy


if __name__ == '__main__':

	results = cv_results(sys.argv[1])
	print(f"{np.mean(results) * 100:.0f}%, ", f"{np.std(results) * 100:.0f}%")

