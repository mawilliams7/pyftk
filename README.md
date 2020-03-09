# pyftk
pyftk is a Python implementation of the fast tree kernel that can be used to generate a normalized similarity matrix batch-wise given a list of serialized parse trees and a specified output path.

pyftk can be used from the command line in the following ways:

python pyftk.py serialized_parse_trees.pkl output_path

python pyftk.py serialized_parse_trees.pkl output_path 565

The first parameter pyftk takes is the filename of the list of serialized NLTK parse trees, the second parameter is the output path, and the third parameter is the specified row offset. The first two parameters are required, and the third is optional. It is important to note that I have included a file that contains a list of serialized NLTK parse trees (50,000 unique parse trees) with pyftk that was generated from sampling sentences scraped from New York Times (NYT) articles published between 2000 - 2016 in the U.S. and World News sections. A sample run of pyftk with the provided input file is provided here:

python pyftk.py nyt_parse_trees.pkl similarity_row

pyftk requires Python 3.7 to be installed to be executed, as well as the following packages:

NLTK
pathlib
pickle
