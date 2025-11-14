from parser import parse
from augmentor import augment

print("PROCESSING RAW DATA")

print("\tparsing...")
parse()
print("\taugmenting...")
augment()

print("DONE")