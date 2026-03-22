'''
用户存储用户的user profile
'''
import json
import os.path
from utils.logger_handler import logger as log
from utils.path_tool import get_abs_path
from utils.config_handler import agents_config
USER_PROFILES = {}

def get_user_profile_path(user_id: str | int) -> str:
    '''
    获取user profile的文件路径
    :param user_id:
    :return:
    '''
    return get_abs_path(f"{agents_config['user_profile_path']}/{user_id}.json")


def build_default_profile():
    return {
        "pets": {
            "cat": None,
            "dog": None
        },
        "budget": None,
        "location": None,
        "preferences": {
            "low_noise": False,
            "easy_maintenance": False
        },
        "home_features": {
            "carpet": False
        }
    }


def save_profile_to_file(profile_path, default_profile):
    try:
        os.makedirs(os.path.dirname(profile_path),exist_ok=True)
        with open(profile_path,"w",encoding="utf-8") as f:
            json.dump(default_profile,f,ensure_ascii=False)
    except Exception as e:
        log.error(f"save profile error:{e}")

def load_profile_from_file(profile_path):
    try:
        with open(profile_path,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error(f"load profile error:{e}")
        return None

def get_user_profile(user_id:str|int) -> dict:
    profile_path = get_user_profile_path(user_id)
    user_profile = load_profile_from_file(profile_path)
    if not user_profile:
        default_profile = build_default_profile()
        save_profile_to_file(profile_path, default_profile)
    USER_PROFILES[user_id] = user_profile
    return user_profile

def update_user_profile(user_id:str|int,patch:dict) -> bool:
    '''
    更新用户的profile
    :param user_id:
    :param profile:
    :return:
    '''
    user_profile = get_user_profile(user_id)
    # 用户不存在
    if not user_profile:
        user_profile = build_default_profile()
        USER_PROFILES[user_id] = user_profile
    updated_user_profile = merge_profile(user_profile,patch)
    USER_PROFILES[user_id] = updated_user_profile
    log.info(f"update user profile:{user_id}  {user_profile}")
    save_profile_to_file(get_user_profile_path(user_id),updated_user_profile)
    return True

def merge_profile(user_profile:dict,patch:dict)->dict:
    '''
    根据patch的内容,更新user_profile
    :param user_profile:
    :param patch:
    :return:
    '''
    for key,value in patch.items():
        if key in user_profile:#key在user_profile中
            if isinstance(user_profile[key],dict):#user_profile[key]是dict,且套字段,则递归调用
                user_profile[key] = merge_profile(user_profile[key],value)
            else:
                user_profile[key] = value
        else:
            user_profile[key] = value

    return user_profile

if __name__ == '__main__':
    get_user_profile("0001")
    print(update_user_profile("0001", {"budget": 3000}))
    print(update_user_profile("0001", {"preferences": {"low_noise": True}}))
    print(update_user_profile("0001", {"pets": {"cat": 2}}))







