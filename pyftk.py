import sys
import pickle
import math
import time
import nltk
from nltk.tree import Tree
from pathlib import Path

# Mutable named constants. Change depending on dataset
FILE_EXTENSION = "_from_tree_kernel.pkl"
OFFSET = 0

all_parse_trees = list()

class Memoize:
	def __init__(self, fn):
		self.fn = fn
		self.memo = {}

	def __call__(self, *args):
		if args not in self.memo:
			self.memo[args] = self.fn(*args)
		return self.memo[args]

def retrieve_parse_trees(parse_tree_filename):
	with open(parse_tree_filename, 'rb') as pkl_file:
		all_parse_trees = pickle.load(pkl_file)
	print("Parse trees retrieved.")
	return all_parse_trees

def extract_production_rules(tree, production_rules):
	left_side = tree.label()
	right_side = ""
	for subtree in tree:
		if type(subtree) == nltk.tree.Tree:
			right_side = right_side + " " + subtree.label()
			extract_production_rules(subtree, production_rules)
		else:
			right_side = right_side + " " + subtree
	production_rules.append((left_side + " ->" + right_side, tree))

def find_node_pairs(first_tree, second_tree):
	node_pairs = set()
	first_tree_production_rules = list()
	extract_production_rules(first_tree, first_tree_production_rules)
	first_tree_production_rules = sorted(first_tree_production_rules, key=lambda x : x[0])
	second_tree_production_rules = list()
	extract_production_rules(second_tree, second_tree_production_rules)
	second_tree_production_rules = sorted(second_tree_production_rules, key=lambda x : x[0])
	node_1 = first_tree_production_rules.pop(0)
	node_2 = second_tree_production_rules.pop(0)
	while node_1[0] != None and node_2[0] != None:
		if node_1[0] > node_2[0]:
			if len(second_tree_production_rules) > 0:
				node_2 = second_tree_production_rules.pop(0)
			else:
				node_2 = [None]
		elif node_1[0] < node_2[0]:
			if len(first_tree_production_rules) > 0:
				node_1 = first_tree_production_rules.pop(0)
			else:
				node_1 = [None]
		else:
			while node_1[0] == node_2[0]:
				second_tree_production_rules_index = 1
				while node_1[0] == node_2[0]:
					node_pairs.add((str(node_1[1]), str(node_2[1])))
					if second_tree_production_rules_index < len(second_tree_production_rules):
						node_2 = second_tree_production_rules[second_tree_production_rules_index]
						second_tree_production_rules_index += 1
					else:
						node_2 = [None]
				if len(first_tree_production_rules) > 0:
					node_1 = first_tree_production_rules.pop(0)
				else:
					node_1 = [None]
				if len(second_tree_production_rules) > 0:
					node_2 = second_tree_production_rules[0]
				else:
					node_2 = [None]
				if node_1[0] == None and node_2[0] == None:
					break
	return node_pairs

@Memoize
def fast_tree_kernel(first_tree_index, second_tree_index):
	global all_parse_trees
	kernel_score = 0
	first_tree = all_parse_trees[first_tree_index]
	second_tree = all_parse_trees[second_tree_index]
	node_pairs = find_node_pairs(first_tree, second_tree)
	for node in node_pairs:
		if node[0] == node[1]:
			kernel_score += 1
	return kernel_score

def normalized_fast_tree_kernel(first_tree_index, second_tree_index):
	return fast_tree_kernel(first_tree_index, second_tree_index) / math.sqrt(fast_tree_kernel(first_tree_index, first_tree_index) * fast_tree_kernel(second_tree_index, second_tree_index))

def compute_similarity_from_tree_kernel(all_parse_trees, offset, output_path):
	rows_computed = offset
	for parse_tree_index in range(len(all_parse_trees)):
		dissimilarity_row = list()
		start_time = time.time()
		for element_index in range(rows_computed + 1, len(all_parse_trees)):
			score = normalized_fast_tree_kernel(parse_tree_index, element_index)
			dissimilarity_row.append(score)
		output_filename = output_path + "_" + str(rows_computed) + FILE_EXTENSION
		with open(output_filename, 'wb') as pkl_file:
			pickle.dump(dissimilarity_row, pkl_file)
		rows_computed += 1
		print("Row " + str(rows_computed) + " computed in: " + (str(time.time() - start_time)))
		del dissimilarity_row
	print("Dissimilarity matrix computed.")

def main():
	global all_parse_trees
	parse_tree_filename = sys.argv[1]
	output_path = sys.argv[2]
	parse_tree_path = Path(parse_tree_filename)

	offset = OFFSET

	if len(sys.argv) > 3:
		offset = max(OFFSET, int(sys.argv[3]))
	else:
		print("Invalid input arguments.")

	if parse_tree_path.exists():
		all_parse_trees = retrieve_parse_trees(parse_tree_filename)
		compute_similarity_from_tree_kernel(all_parse_trees, offset, output_path)
	else:
		print("Invalid parse tree path.")

if __name__ == '__main__':
	main()
