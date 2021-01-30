# script to seperate out train and test corpus into different folders
# read data from train_files.txt and test_files.txt and put them into different folders
# we could even soft link them to save memory

import shutil

with open('data/scdb/test_files.txt') as test_files:
  test_list = test_files.readlines()

test_list = [t.strip() for t in test_list]
print(test_list[:10])

dst_dir = 'data/scdb/raw_test/'
for src in test_list:
  dst = dst_dir + src.split('/')[-1]
  print(dst)
  shutil.copyfile(src, dst)
