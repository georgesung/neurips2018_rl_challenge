import os
from gstorage import download_blob, find_latest_checkpoint

NUM_CFGS = 12
BUCKET = 'your-bucket'
EXP_OUT_BASE = 'exp_1024/exp_output/'

for i in range(NUM_CFGS):
    local_out_dir = 'exp_out/cfg%d' % i
    remote_out_dir = os.path.join(EXP_OUT_BASE, 'cfg%d' % i)

    # Local output directory for experiment artifacts
    if not os.path.isdir(local_out_dir):
        os.system('mkdir -p %s' % local_out_dir)

    # Load latest agent from gcloud storage
    # Find latest checkpoint file in output_dir
    extra_data_file, tune_metadata_file = find_latest_checkpoint(
        bucket_name=BUCKET,
        checkpoint_dir=remote_out_dir)

    # Only restore previous agent if there was a previous checkpoint
    if extra_data_file is not None:
        print('Downloading files for cfg%d' % i)

        # Download tf checkpoint files
        download_blob(
            bucket_name=BUCKET,
            source_blob_name=extra_data_file,
            destination_file_name=os.path.join(local_out_dir, os.path.basename(extra_data_file)))
        download_blob(
            bucket_name=BUCKET,
            source_blob_name=tune_metadata_file,
            destination_file_name=os.path.join(local_out_dir, os.path.basename(tune_metadata_file)))

        # Download logs
        download_blob(
            bucket_name=BUCKET,
            source_blob_name=os.path.join(remote_out_dir, 'best_agents.log'),
            destination_file_name=os.path.join(local_out_dir, 'best_agents.log'))
        download_blob(
            bucket_name=BUCKET,
            source_blob_name=os.path.join(remote_out_dir, 'train.log'),
            destination_file_name=os.path.join(local_out_dir, 'train.log'))
        download_blob(
            bucket_name=BUCKET,
            source_blob_name=os.path.join(remote_out_dir, 'train_verbose.log'),
            destination_file_name=os.path.join(local_out_dir, 'train_verbose.log'))
    else:
        print('WARNING: cfg%d has no checkpoint in remote' % i)
