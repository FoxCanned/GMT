import os
import shutil

# 68.
source_folder = "/home/zyh/data/TrackEval/res"  # 原始文件夹路径
destination_folder = "/home/zyh/data/TrackEval/vision"  # 目标文件夹路径
# 创建目标文件夹
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍历原始文件夹中的所有文件夹
for folder_name in os.listdir(source_folder):
    folder_path = os.path.join(source_folder, folder_name)
    # 检查是否是文件夹
    if os.path.isdir(folder_path):
        # 遍历文件夹中的文件

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # 检查文件是否是txt文件
            if filename.endswith(".txt"):
                # 获取视图号（View1或View2）
                #vision
                #view_number = filename.split("View")[1].split(".")[0]
                view_number = filename.split(".")[0]


                # 生成目标文件名
                #vision
                #new_filename = f"{folder_name}_View{view_number}.txt"
                new_filename = f"{folder_name}_{view_number}.txt"
                new_file_path = os.path.join(destination_folder, new_filename)
                # 复制文件到目标文件夹中
                shutil.copy(file_path, new_file_path)