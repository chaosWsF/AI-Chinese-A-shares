import os
import glob
import shutil

dir1s = glob.glob('D:/rongshidata/experiment_data_1/monthly/end/feature_v*')
for dir1 in dir1s:
    dir2s = glob.glob(dir1 + '/prob/*m_*_*')
    for dir2 in dir2s:
        dir3s = glob.glob(dir2 + '/PNE*_NL_*')
        for dir3 in dir3s:
            for file in os.listdir(dir3):
                if 'prob' in file:
                    continue

                path = dir3 + '/' + file

                try:
                    os.remove(path)
                except OSError:
                    shutil.rmtree(path)

        print(dir2, 'DONE')
