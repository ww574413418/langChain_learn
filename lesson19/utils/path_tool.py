'''
为整个项目提供统一的绝对路径
'''
import os

def get_project_root() -> str:
    # 当前文件的绝对路径 "/Users/grubby/.../lesson19/utils/path_tool.py"
    current_file_absulute_path = os.path.abspath(__file__)
    # 获取当前文件的目录 "/Users/grubby/.../lesson19/utils"
    current_dir = os.path.dirname(current_file_absulute_path)
    # 获取当前文件夹的目录 "/Users/grubby/.../lesson19"
    project_root = os.path.dirname(current_dir)

    return project_root

def get_abs_path(relative_path:str):
    '''
    传入相对路径,得到绝对路径
    :param relative_path:
    :return:
    '''
    # 拿到根目录
    project_root = get_project_root()
    return os.path.join(project_root,relative_path)

if __name__ == '__main__':
    print(get_abs_path("data/扫地机器人100问.pdf"))