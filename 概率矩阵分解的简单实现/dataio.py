import os
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, np.int64):
			return int(obj)
		else:
			return super(NpEncoder, self).default(obj)

def prepare_folder(path):
	"""Create folder from path."""
	if not os.path.isdir(path):
		os.makedirs(path)

def build_new_paths(DATA_FOLDER,DATASET_NAME):
	"""Create dataset folder path name."""
	CSV_FOLDER = os.path.join(DATA_FOLDER, DATASET_NAME)
	return CSV_FOLDER

def SaveDict(dictionary,folder,data_name):
	folder_path = os.path.join(str(folder), str(data_name))
	string=json.dumps(dictionary,indent=2,cls=NpEncoder)
	with open(folder_path,'w')as f:
		f.write(string)
	f.close()

def LoadDict(folder,data_name):
	folder_path = os.path.join(str(folder), str(data_name))
	string = open(folder_path)
	return json.load(string)

def ReverseDict(dictionary):
	return dict(zip(dictionary.values(), dictionary.keys()))
