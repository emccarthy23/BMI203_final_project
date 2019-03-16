import sys
from .io import read_sequences, reverse_complement, read_neg_sequences, Diff, convert_to_numeric
import neural_net

# Some quick stuff to make sure the program is called correctly

# if len(sys.argv) != 4:
#     print("Usage: python -m BMI203_final_project [-A | -T] <path_pos_seq>  <path_neg_seq>")
#     sys.exit(0)
