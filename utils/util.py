import os
import pickle

from utils.dist import get_rank, get_world_size, synchronize, is_master


def fetch_data_by_disk(data, path):
    assert isinstance(data, dict)
    
    final_path = path
    
    root = os.path.split(path)[0]
    if is_master() and not os.path.exists(root):
        os.makedirs(root)
    synchronize()

    path, ext = os.path.splitext(path)
    tmp_file_path = path + f'_rank{get_rank()}' + ext
    
    pickle.dump(data, open(tmp_file_path, 'wb'))

    synchronize()
    
    if is_master():
        collect_result = dict()
        for i in range(get_world_size()):
            tmp_file_path = path + f'_rank{i}' + ext
            data = pickle.load(open(tmp_file_path, 'rb'))
            collect_result.update(data)
            os.remove(tmp_file_path)
    
        pickle.dump(collect_result, open(final_path, 'wb'))
    
    synchronize()
    
    collect_result = pickle.load(open(final_path, 'rb'))

    return collect_result
    

    