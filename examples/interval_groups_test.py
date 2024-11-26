from stpy.helpers.helper import interval_groups, get_hierarchy, hierarchical_distance, valid_enlargement

if __name__ == "__main__":

	out = get_hierarchy(start = 0,new_elements=[1,2,3])
	curr =  [[0], [1], [2], [3]]
	print(hierarchical_distance(curr, [[0,1],[2],[3]]))
	enlargements = valid_enlargement(curr, out)
	for enlargement in enlargements:
		print (curr,"->",out[enlargement])