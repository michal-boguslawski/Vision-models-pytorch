import wget
import os
import zipfile

directory_name = 'data'
links = {'train': 'http://images.cocodataset.org/zips/train2017.zip',
         'val': 'http://images.cocodataset.org/zips/val2017.zip',
         'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'}

try:
    os.mkdir(directory_name)
    print(f"Directory '{directory_name}' created successfully.")
except FileExistsError:
    print(f"Directory '{directory_name}' already exists.")
except Exception as e:
    print(f"An error occurred: {e}")

local_file_paths = []
for name, url in links.items():
    print(f"Name: {name}, url: {url}")
    try:
        local_file_path = wget.download(url, out=f"{directory_name}/{name}.zip")
        print(f"\nFile downloaded successfully to '{local_file_path}'")
        local_file_paths.append(local_file_path)
        try:
            with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
                extract_to_directory = f"{directory_name}/"
                zip_ref.extractall(extract_to_directory)
                print(f"Files extracted successfully to '{extract_to_directory}'")
        except zipfile.BadZipFile:
            print("Error: The file is not a valid ZIP archive.")
        except Exception as e:
            print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
