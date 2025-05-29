import os
import torch

if __name__ == '__main__':
    chunks_path = r'D:\crystal_datasets\ellipsoid_chunks'
    os.chdir(chunks_path)
    files = os.listdir(chunks_path)
    files = [file for file in files if "chunk" in file]
    results_dict = torch.load(files[0])

    for file in files[1:]:
        new_results_dict = torch.load(file)
        for key in results_dict.keys():
            results_dict[key] = torch.cat([results_dict[key], new_results_dict[key]])

    print(f"finished processing dataset with length {len(results_dict[key])}")
    torch.save(results_dict, 'combined_ellipsoid_dataset.pt')
