import numpy as np
from scipy import sparse


def MapFileUpload(filenames, opsysWind = False):
	if opsysWind:
		filepath = "C:/Users/Chaotica/Dropbox/Science/PhD/Orsay/Measurements/ttcApril2018/ttcfilm200418/"
	else:
		filepath ="/home/kamila/Dropbox/Science/PhD/Orsay/Measurements/ttcApril2018/ttcfilm200418/"

	files = [filepath + file for file in filenames]

	dict_fieldfreq = {}
	needed_frequencies = []
	needed_fields = []
	datapoints_num = []
	for file in files:
		data = np.loadtxt(file, comments='%', usecols=(0, 1, 9))
		intensity = data[:, 1]
		inten_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
		b_int = sparse.csr_matrix(inten_norm > 0.5)
		intensity_rest = b_int.astype(int)
		_, needed_indeces = intensity_rest.nonzero()
		datapoints_num.append(len(needed_indeces))

		for i in needed_indeces:
			field = round(data[i, 2], 3) * 1e4
			frequency = round((data[i, 0] * 1e-6), 3)
			needed_fields.append(field)
			needed_frequencies.append(frequency)

	for field in needed_fields:
		assigned_frequencies = []
		for i in range(len(needed_fields)):
			if field == needed_fields[i]:
				assigned_frequencies.append(needed_frequencies[i])
		dict_fieldfreq[field] = assigned_frequencies


	return sum(datapoints_num), dict_fieldfreq