def contains(query:str,keywords:list[str]):
    for keyword in keywords:
        if keyword in query:
            return True
    return False

def recall_profile_for_query(query:str,user_profile:dict):
    recalled = {}

    pets = user_profile.get("pets",{})
    budget = user_profile.get("budget")
    location = user_profile.get("location")
    preferences = user_profile.get("preferences",{})
    home_features = user_profile.get("home_features",{})
    other_facts = user_profile.get("other_facts",[])

    if contains(query,["宠物", "猫", "狗", "掉毛", "毛发"]):
        pet_info = {}
        if pets.get("cat") is not None:
            pet_info["cat"] = pets.get("cat")
        if pets.get("dog") is not None:
            pet_info["dog"] = pets.get("dog")

        if pet_info:
            recalled["pets"] = pet_info

    if contains(query,["预算", "价格", "多少钱", "便宜", "贵"]):
        if budget is not None:
            recalled["budget"] = budget

    if contains(query,["天气", "城市", "潮湿", "南方", "北方", "回南天"]):
        if location is not None:
            recalled["location"] = location

    if contains(query,["噪音", "安静"]):
        if preferences.get("low_noise"):
            recalled.setdefault("preferences",{})["low_noise"] = True

    if contains(query, ["地毯"]):
        if home_features.get("carpet"):
            recalled.setdefault("home_features", {})["carpet"] = True

    if contains(query, ["阳台", "落灰", "灰尘"]):
        if home_features.get("balcony"):
            recalled.setdefault("home_features", {})["balcony"] = True
        if other_facts:
            recalled["other_facts"] = other_facts

    if contains(query, ["推荐", "选购", "适合", "哪款"]):
        if budget is not None:
            recalled["budget"] = budget
        if location:
            recalled["location"] = location
        if pets.get("cat") is not None or pets.get("dog") is not None:
            pet_info = {}
            if pets.get("cat") is not None:
                pet_info["cat"] = pets["cat"]
            if pets.get("dog") is not None:
                pet_info["dog"] = pets["dog"]
            recalled["pets"] = pet_info
        if preferences.get("low_noise"):
            recalled.setdefault("preferences", {})["low_noise"] = True
        if preferences.get("easy_maintenance"):
            recalled.setdefault("preferences", {})["easy_maintenance"] = True
        if home_features.get("carpet"):
            recalled.setdefault("home_features", {})["carpet"] = True
        if home_features.get("balcony"):
            recalled.setdefault("home_features", {})["balcony"] = True

    return  recalled

def format_recalled_profile(recalled_profile:dict)->str:
    if not recalled_profile:
        return ""

    lines = ["当前用户的长期信息:"]

    pets = recalled_profile.get("pets",{})
    pets_parts = []
    if pets.get("cat") is not None:
        pets_parts.append(f"{pets.get('cat')}只猫")
    if pets.get("dog") is not None:
        pets_parts.append(f"{pets.get('dog')}只狗")
    if pets_parts:
        lines.append(f"宠物: {', '.join(pets_parts)}")

    budget = recalled_profile.get("budget")
    if budget is not None:
        lines.append(f"预算: {budget}")
    location = recalled_profile.get("location")
    if location is not None:
        lines.append(f"城市: {location}")

    preferences = recalled_profile.get("preferences",{})
    pref_parts = []
    if preferences.get("low_noise"):
        pref_parts.append("低噪音")
    if preferences.get("easy_maintenance"):
        pref_parts.append("易维护")
    if pref_parts:
        lines.append(f"偏好: {', '.join(pref_parts)}")
    home_features = recalled_profile.get("home_features", {})
    env_parts = []
    if home_features.get("carpet"):
        env_parts.append("有地毯")
    if home_features.get("balcony"):
        env_parts.append("有阳台")
    if env_parts:
        lines.append(f"- 家庭环境：{'、'.join(env_parts)}")

    other_facts = recalled_profile.get("other_facts", [])
    if other_facts:
        lines.append(f"- 其他长期信息：{'；'.join(other_facts)}")

    return "\n".join(lines)