import os

input_dir = "/home/khanhtty/dataset/person_face_sat_coco/images/sat_train"
output_file = "/home/khanhtty/dataset/person_face_sat_coco/train.txt"
base_replace = "/home/khanhtty/dataset/person_face_sat_coco/images/sat_train"
target_prefix = "./images/sat_train"

with open(output_file, "w") as f:
    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            if file.lower().endswith(".jpg"):
                abs_path = os.path.join(root, file)
                rel_path = abs_path.replace(base_replace, target_prefix)
                f.write(rel_path + "\n")

print(f"Wrote paths to {output_file}")
