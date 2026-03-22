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

class UserProfilePatch(BaseModel):
    pets:Optional[PetPatch] = None
    budget:Optional[int] = Field(default=None,description="预算")
    location:Optional[str] = Field(default=None,description="位置")
    preferences:Optional[PreferencePatch] = None
    home_features:Optional[HomeFeaturePatch] = None

def extractor_userProfile_patch(query: str) -> dict:
    structured_llm = chat_model.with_structured_output(UserProfilePatch)
    prompt = f"{extract_userprofile_prompt()} \n\n用户输入:{query}"
    result = structured_llm.invoke(prompt)
    log.info(f"extract_userProfile_patch:{result}")
    return result.model_dump(exclude_none=True)

if __name__ == '__main__':
    print(extractor_userProfile_patch("我有一只猫,2条狗,住在北京"))