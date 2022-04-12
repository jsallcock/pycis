try:
	from .flow_matrix import FlowGeoMatrix
except Exception as e:
	print('WARNING: pycis.inversions is unavailable due to error: {0}'.format(e))


