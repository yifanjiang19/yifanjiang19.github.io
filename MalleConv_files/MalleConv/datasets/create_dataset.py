from datasets.large_patch_noclip import large_patch_noclip_dataset

def build_dataset(dataset,
                  patch_size,
                  batch_size,
                  sigma,
                  noise_variance,
                  cell,
                  mode
                  ):
  ds_train, ds_valid = large_patch_noclip_dataset(patch_size,
                                             batch_size,
                                             sigma,
                                             10000,
                                             noise_variance,
                                             cell)
  return ds_train, ds_valid
