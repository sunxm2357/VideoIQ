import os
import os.path as osp
import argparse
import lmdb
import pyarrow as pa
import concurrent.futures
import time
import numpy as np

parser = argparse.ArgumentParser(description='Create video (images) into lmdb database. '
                                             'Current logic is put a video into one transaction')
parser.add_argument('-d', '--data_path', help='location of input video folder.', type=str)
parser.add_argument('-s', '--set', type=str, help='which set of videos',
                    choices=['train', 'val', 'test', 'mini_train', 'mini_val', 'tiny_train', 'tiny_val'])
parser.add_argument('--sep', help='separator', type=str, default=' ')
parser.add_argument('-n', '--num_processes', type=int, help='number of processes', default=36)
parser.add_argument('-vr', '--video_range', default=[], type=int, nargs="+", help='video range')

args = parser.parse_args()

def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()

MAX_FRAMES = 1000

def load_a_video(directory, video, chunk_id, num_chunks=-1):
    vid_id = video[0].split("/")[-1]
    label = int(video[-1])
    img_prefix = osp.join(directory, video[0])

    start_frame_idx = int(video[1])
    end_frame_idx = int(video[2])
    num_frames = end_frame_idx - start_frame_idx + 1

    #redefine start and end idx
    if chunk_id != -1:
        start_frame_idx = MAX_FRAMES * chunk_id + 1
        end_frame_idx = min(end_frame_idx, MAX_FRAMES * (chunk_id + 1))

    imgs = ()
    if chunk_id == -1 or chunk_id == 0:
        imgs = imgs + (num_frames,)
    for idx in range(start_frame_idx, end_frame_idx + 1):
        compressed_img = open(osp.join(img_prefix, "{:05d}.jpg".format(idx)), 'rb').read()
        imgs = imgs + (compressed_img,)
    if chunk_id == -1 or chunk_id == num_chunks - 1:
        imgs = imgs + (label,)

    return imgs, vid_id, chunk_id, num_chunks

def create_video_lmdb(write_frequency=1000):
    directory = args.data_path
    file_list = osp.join(directory, args.set + ".txt")
    print("Loading dataset from %s" % file_list)

    # parse the list
    videos = []
    with open(file_list) as f:
        for line in f.readlines():
            line = line.strip()
            if line != "":
                videos.append(line.split(args.sep))

    vid_ids = [video[0].split("/")[-1] for video in videos]


    if len(args.video_range) != 0:
        if len(args.video_range) != 2:
            raise ValueError("only specify start and end")
    print("From {} to {}".format(args.video_range[0], args.video_range[1]))
    videos = videos[args.video_range[0]: args.video_range[1]]

    # create lmdb
    lmdb_path = osp.join(directory, "%s.lmdb" % args.set)
    isdir = os.path.isdir(lmdb_path)
    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        futures = []
        total_videos = len(videos)
        for video in videos:
            start_frame_idx = int(video[1])
            end_frame_idx = int(video[2])
            num_frames = end_frame_idx - start_frame_idx + 1
            num_chunks = int(np.ceil(num_frames / MAX_FRAMES))
            if num_chunks == 1:
                futures.append(executor.submit(load_a_video, directory, video, -1))
            else:
                for i in range(num_chunks):
                    futures.append(executor.submit(load_a_video, directory, video, i, num_chunks))

        vid_idx = 0
        tmp_chunk_vid = {}
        start_time = time.time()
        for future in futures:
            imgs, vid_id, chunk_id, num_chunks = future.result()
            if chunk_id == -1:
                txn.put(u'{}'.format(vid_id).encode('ascii'), dumps_pyarrow(imgs))
                if vid_idx % write_frequency == 0:
                    speed = time.time() - start_time
                    start_time = time.time()
                    print("[{}/{}] Speed: {:.3f} sec".format(vid_idx, total_videos, speed),
                          flush=True)
                    txn.commit()
                    txn = db.begin(write=True)
                vid_idx += 1
            else:
                if vid_id not in tmp_chunk_vid:
                    tmp_chunk_vid[vid_id] = imgs
                else:
                    tmp_chunk_vid[vid_id] += imgs
                if len(tmp_chunk_vid[vid_id]) - 2 == tmp_chunk_vid[vid_id][0]:
                    txn.put(u'{}'.format(vid_id).encode('ascii'), dumps_pyarrow(tmp_chunk_vid[vid_id]))
                    if vid_idx % write_frequency == 0:
                        speed = time.time() - start_time
                        start_time = time.time()
                        print("[{}/{}] Speed: {:.3f} sec".format(vid_idx, total_videos, speed),
                              flush=True)
                        txn.commit()
                        txn = db.begin(write=True)
                    vid_idx += 1
                    tmp_chunk_vid.pop(vid_id, None)
    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in vid_ids]
    with db.begin(write=True) as txn:
        if txn.get(b'__keys__', default=None) is None:
            txn.put(b'__keys__', dumps_pyarrow(keys))
        if txn.get(b'__len__', default=None) is None:
            txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    create_video_lmdb()