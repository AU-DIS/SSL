import os

def delete_files_with_ged(path):
    for root, _, files in os.walk(path):
        for file in files:
            if 'ged' in file:
                os.remove(os.path.join(root, file))

delete_files_with_ged('experiments_final/')
