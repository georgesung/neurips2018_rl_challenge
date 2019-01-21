from google.cloud import storage
import argparse
import re

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """
    Downloads a blob from the bucket
    https://cloud.google.com/storage/docs/downloading-objects#storage-download-object-python
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to the bucket
    https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

def find_latest_checkpoint(bucket_name, checkpoint_dir):
    """
    Find the latest checkpoint file
    Returns tuple of (extra_data_file, tune_metadata_file)
    """
    # Google Storage API not smart enough, need to end the checkpoint_dir with '/'
    if checkpoint_dir[-1] != '/':
        checkpoint_dir += '/'

    # Now we can find the latest checkpoint files
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=checkpoint_dir, delimiter=None)

    latest_checkpoint_ep = -1
    latest_checkpoint_prefix = ''

    for blob in blobs:
        file_name = blob.name
        match = re.search(r'^(.+/checkpoint-([0-9]+))\.extra_data', file_name)
        if not match:
            continue
        checkpoint_ep = int(match.group(2))
        checkpoint_prefix = match.group(1)

        if checkpoint_ep > latest_checkpoint_ep:
            latest_checkpoint_ep = checkpoint_ep
            latest_checkpoint_prefix = checkpoint_prefix

    if latest_checkpoint_prefix == '':
        return (None, None)
    else:
        return (latest_checkpoint_prefix + '.extra_data', latest_checkpoint_prefix + '.tune_metadata')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download/upload from/to Google Cloud Storage')
    parser.add_argument('--download', dest='download', action='store_true', default=True)
    parser.add_argument('--upload', dest='download', action='store_false', default=True)
    parser.add_argument('--bucket', dest='bucket', action='store', default=None)
    parser.add_argument('--src', dest='src', action='store', default=None)
    parser.add_argument('--dst', dest='dst', action='store', default=None)
    args = parser.parse_args()

    if args.bucket is None or args.src is None or args.dst is None:
        raise ValueError('Wrong args')

    if args.download:
        download_blob(args.bucket, args.src, args.dst)
    else:
        upload_blob(args.bucket, args.src, args.dst)
