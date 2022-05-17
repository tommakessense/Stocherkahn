from os import path, listdir, walk
from pathlib import Path
from typing import Set

import cv2

from utils import imp_args


@imp_args
def get_args(parser):
    parser.add_argument("datadir")
    parser.add_argument("--img-sec", type=int, default=1)


def get_extracted(img_dir) -> Set:
    _extracted = set({})
    for root, dirs, files in walk(img_dir):
        for file in files:
            _path = Path(f"{root}/{file}")
            video = file.split('__')
            if len(video) < 2:
                continue
            _extracted.add(video[0])
    return _extracted


def process_video_dir(data_dir, extracted):
    vid_dir = path.join(data_dir, 'videos')
    dest_dir = path.join(data_dir, 'labeled')
    videos = listdir(vid_dir)
    for video in videos:
        full_path = path.join(data_dir, 'videos', video)
        path_obj = Path(full_path)
        if path_obj.stem in extracted:
            print(f"video {path_obj.name} already extracted")
            continue
        extract_images(path_obj, dest_dir)
    print(f"{len(videos)} videos processed")


def extract_images(path_obj: Path, dest_dir, img_per_sec=1):
    name = path_obj.stem
    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        print(f"Error: could not open {path_obj.name}")
        return
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0 else 30
    count = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        if count % fps//img_per_sec == 0:
            img_name = path.join(dest_dir, f"{name}__{img_per_sec}__{count}.jpg")
            cv2.imwrite(img_name, image)  # save frame as JPEG file
            print(f"added image {img_name}")
        count += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_args()
    datadir = args.datadir
    images_per_sec = args.img_sec

    extracted = get_extracted(path.join(datadir, 'labeled'))
    extracted_tests = get_extracted(path.join(datadir, 'tests'))
    process_video_dir(datadir, extracted.union(extracted_tests))
