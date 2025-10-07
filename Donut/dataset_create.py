import json
from datasets import Dataset, Image
import os
import shutil

def prepare_dataset_dict(json_file, images_dir):
  ds_dict = {
    "image": [],
    "ground_truth": []
  }

  for single_json in json_file:
    if single_json["aug"] == []:
      image_name = single_json['image_code']
      ds_dict["image"].append(f"{images_dir}/{image_name}")
      json_str = json.dumps(single_json)
      ds_dict["ground_truth"].append(json_str)
    else:
      for aug in single_json["aug"]:
        image_name = aug + "_" + single_json['image_code']
        ds_dict["image"].append(f"{images_dir}/{image_name}")
        json_str = json.dumps(single_json)
        ds_dict["ground_truth"].append(json_str)
  return ds_dict

def create_dataset(json_file_path, images_dir, output_dir):
  with open(json_file_path, encoding='utf8') as file:
    json_file = json.load(file)
  out_dict = prepare_dataset_dict(json_file, images_dir)
  dataset = Dataset.from_dict(out_dict).cast_column("image", Image())
  dataset.save_to_disk(output_dir)
  print(f"Dataset saved to {output_dir}")

def remove_dataset(output_dir):
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print(f"Removed dataset at {output_dir}")
  else:
    print(f"Dataset at {output_dir} does not exist.")

if __name__ == "__main__":
  create_dataset(
    json_file_path="/home/dyplom/Dataset/my_dataset/test/donut_data.json",
    images_dir="/home/dyplom/Dataset/my_dataset/test/images",
    output_dir="/home/dyplom/Donut/datasets/test"
  )

  create_dataset(
    json_file_path="/home/dyplom/Dataset/my_dataset/val/donut_data.json",
    images_dir="/home/dyplom/Dataset/my_dataset/val/images",
    output_dir="/home/dyplom/Donut/datasets/val"
  )

  create_dataset(
    json_file_path="/home/dyplom/Dataset/my_dataset/train/donut_data.json",
    images_dir="/home/dyplom/Dataset/my_dataset/train/images",
    output_dir="/home/dyplom/Donut/datasets/train"
  )
