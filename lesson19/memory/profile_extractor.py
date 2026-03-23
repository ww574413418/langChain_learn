from typing import Optional
from pydantic import BaseModel,Field
from model.model_factory import chat_model
from utils.prompts_loader import extract_userprofile_prompt
from utils.logger_handler import logger as log

class PetPatch(BaseModel):
    cat: Optional[ int] = Field(default=None,description="猫的数量,未知则为null")
    dog: Optional[ int] = Field(default=None,description="狗的数量,未知则为null")

class PreferencePatch(BaseModel):
    low_noise: Optional[ bool] = Field(default=None,description="是否需要低噪音")
    easy_maintenance: Optional[ bool] = Field(default=None,description="是否需要易维护")

class HomeFeaturePatch(BaseModel):
    carpet: Optional[ bool] = Field(default=None,description="家里是否有地毯")
    balcony: Optional[bool] = Field(default=None,description="是否有阳台")

class UserProfilePatch(BaseModel):
    pets:Optional[PetPatch] = None
    budget:Optional[int] = Field(default=None,description="预算")
    location:Optional[str] = Field(default=None,description="位置")
    preferences:Optional[PreferencePatch] = None
    home_features:Optional[HomeFeaturePatch] = None
    other_facts: Optional[list[str]] = None

def prune_empty_fields(data):
    '''
    取除空套壳
    把
    {
    "pets": {},
    "preferences": {},
    "home_features": {}
    }
    变成{}
    :param data:
    :return:
    '''
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            pruned_value = prune_empty_fields(value)
            if pruned_value not in (None, {}, [], ""):
                cleaned[key] = pruned_value
        return cleaned

    if isinstance(data, list):
        cleaned = [prune_empty_fields(item) for item in data]
        return [item for item in cleaned if item not in (None, {}, [], "")]

    return data

def extractor_userProfile_patch(query: str,current_profile: dict) -> dict:
    structured_llm = chat_model.with_structured_output(UserProfilePatch)
    prompt = f"{extract_userprofile_prompt()} \n\n用户输入:{query} \n\n 当前用户信息:{current_profile}"
    result = structured_llm.invoke(prompt)
    patch = result.model_dump(exclude_none=True)
    patch = prune_empty_fields(patch)
    log.info(f"extract_userProfile_patch:{result}")
    return patch

if __name__ == '__main__':
    print(extractor_userProfile_patch("我有一只猫,2条狗,住在北京"))