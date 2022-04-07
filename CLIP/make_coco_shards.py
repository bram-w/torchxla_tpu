import sys
import os
import os.path
import random
import argparse

from torchvision import datasets

import webdataset as wds


parser = argparse.ArgumentParser("""Generate sharded dataset from original CocoCaptions data.""")
parser.add_argument("--splits", default="train,val", help="which splits to write")
parser.add_argument(
    "--filekey", action="store_true", help="use file as key (default: index)"
)
parser.add_argument("--maxsize", type=float, default=1e9)
parser.add_argument("--maxcount", type=float, default=1000)
parser.add_argument(
    "--shards", default="./coco_shards", help="directory where shards are written"
)
parser.add_argument(
    "--img_data",
    default="/export/share/datasets/vision/coco/images/",
    help="directory containing CocoCaptions data distribution suitable for torchvision.datasets",
)
parser.add_argument(
    "--ann_data",
    default="/export/share/datasets/vision/coco/annotations/",
    help="directory containing CocoCaptions data distribution suitable for torchvision.datasets",
)
parser.add_argument("--year", type=int, default=2017)
args = parser.parse_args()


assert args.maxsize > 10000000
assert args.maxcount < 1000000


if not os.path.isdir(f"{args.img_data}/train{args.year}"):
    print(f"{args.data}: should be directory containing CocoCaptions", file=sys.stderr)
    print(f"suitable as argument for torchvision.datasets.CocoCaptions(...)", file=sys.stderr)
    sys.exit(1)

try:
  os.mkdir(args.shards)
except:
  assert not len(os.listdir(args.shards))

splits = args.splits.split(",")


def readfile(fname):
    "Read a binary file from disk."
    with open(fname, "rb") as stream:
        return stream.read()


all_keys = set()

def _load_image(self, id):
  return self.coco.loadImgs(id)[0]["file_name"]
datasets.CocoCaptions._load_image = _load_image

def write_dataset(img_data, ann_data, split, year=2017, base="./shards"):

    # We're using the torchvision CocoCaptions dataset
    # to parse the metadata; however, we will read
    # the compressed images directly from disk (to
    # avoid having to reencode them)
    img_data = f"{img_data}/{split}{year}/"
    ann_data = f"{ann_data}/captions_{split}{year}.json"
    ds = datasets.CocoCaptions(img_data, ann_data)
    nimages = len(ds)
    print("# nimages", nimages)

    # We shuffle the indexes to make sure that we
    # don't get any large sequences of a single class
    # in the dataset.
    indexes = list(range(nimages))
    random.shuffle(indexes)

    # This is the output pattern under which we write shards.
    pattern = os.path.join(base, f"coco-captions-{split}-%06d.tar")

    with wds.ShardWriter(pattern, maxsize=int(args.maxsize), maxcount=int(args.maxcount)) as sink:
        for i in indexes:

            # Internal information from the CocoCaptions dataset
            # instance: the file name and the numerical class.
            jpeg_name, txt_list = ds[i]
            txt = "CAPTIONBREAK".join(txt_list)
            fname = f"{img_data}/{jpeg_name}"
            # Read the JPEG-compressed image file contents.
            image = readfile(fname)

            # Construct a uniqu keye from the filename.
            key = os.path.splitext(os.path.basename(fname))[0]

            # Useful check.
            assert key not in all_keys
            all_keys.add(key)

            # Construct a sample.
            xkey = key if args.filekey else "%07d" % i
            sample = {"__key__": xkey, "jpg": image, "txt": txt}

            # Write the sample to the sharded tar archives.
            sink.write(sample)


for split in splits:
    print("# split", split)
    write_dataset(args.img_data, args.ann_data, base=args.shards, split=split, year=args.year)
