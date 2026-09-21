"""提示词增强模块 - 基于同义词扩展"""

from typing import List, Dict, Set


class PromptEnhancer:
    """
    提示词增强器 - 通过同义词扩展增强检测效果

    支持通用词库和遥感领域专用词库
    """

    # 通用词库 - 常见目标检测类别
    GENERAL_MAPPINGS: Dict[str, List[str]] = {
        "person": ["person", "people", "human", "pedestrian", "man", "woman", "child", "people walking"],
        "car": ["car", "vehicle", "automobile", "sedan", "suv", "truck", "van", "motor vehicle", "motorcar"],
        "bicycle": ["bicycle", "bike", "cycle", "cycling", "cyclist"],
        "motorcycle": ["motorcycle", "motorbike", "scooter", "moped"],
        "bus": ["bus", "minibus", "coach", "public transport"],
        "truck": ["truck", "lorry", "delivery truck", "semi-truck", "cargo truck"],
        "train": ["train", "railway", "locomotive", "railroad train"],
        "boat": ["boat", "vessel", "ship", "ferry", "sailing boat"],
        "airplane": ["airplane", "aircraft", "plane", "jet", "aeroplane"],
        "dog": ["dog", "puppy", "canine", "retriever"],
        "cat": ["cat", "kitten", "feline"],
        "bird": ["bird", "pigeon", "seagull", "crow", "sparrow"],
        "horse": ["horse", "pony", "equestrian"],
        "sheep": ["sheep", "lamb", "mutton"],
        "cow": ["cow", "cattle", "bull", "ox", "bovine"],
        "elephant": ["elephant", "pachyderm"],
        "bear": ["bear", "grizzly", "polar bear"],
        "zebra": ["zebra", "equine"],
        "giraffe": ["giraffe", "camelopard"],
        "backpack": ["backpack", "bag", "rucksack", "knapsack"],
        "umbrella": ["umbrella", "parasol", "sunshade"],
        "handbag": ["handbag", "purse", "bag", "tote bag"],
        "suitcase": ["suitcase", "luggage", "travel bag", "trunk"],
        "tie": ["tie", "necktie", "bow tie"],
        "skis": ["skis", "skiing equipment"],
        "snowboard": ["snowboard", "snowboarding"],
        "sports ball": ["ball", "sports ball", "football", "basketball"],
        "kite": ["kite", "kiting"],
        "baseball bat": ["baseball bat", "bat"],
        "baseball glove": ["baseball glove", "glove", "mitt"],
        "skateboard": ["skateboard", "skate"],
        "surfboard": ["surfboard", "surf"],
        "tennis racket": ["tennis racket", "racket", "racquet"],
        "wine glass": ["wine glass", "glass", "glass cup"],
        "cup": ["cup", "mug", "glass"],
        "fork": ["fork", "utensil"],
        "knife": ["knife", "blade", "cutlery"],
        "spoon": ["spoon", "utensil"],
        "bowl": ["bowl", "dish", "basin"],
        "banana": ["banana", "fruit"],
        "apple": ["apple", "fruit"],
        "sandwich": ["sandwich", "food"],
        "orange": ["orange", "fruit", "citrus"],
        "broccoli": ["broccoli", "vegetable"],
        "carrot": ["carrot", "vegetable"],
        "hot dog": ["hot dog", "food", "sausage"],
        "pizza": ["pizza", "food"],
        "donut": ["donut", "doughnut", "pastry"],
        "cake": ["cake", "dessert", "pastry"],
        "chair": ["chair", "seat", "seating"],
        "couch": ["couch", "sofa", "settee"],
        "dining table": ["dining table", "table", "desk"],
        "potted plant": ["potted plant", "plant", "flower", "houseplant"],
        "bed": ["bed", "mattress"],
        "mirror": ["mirror", "reflection"],
        # "dining table" merged above (includes "desk")
        "window": ["window", "glass"],
        "laptop": ["laptop", "computer", "notebook"],
        "mouse": ["mouse", "computer mouse"],
        "remote": ["remote", "remote control", "controller"],
        "keyboard": ["keyboard", "computer keyboard"],
        "cell phone": ["cell phone", "phone", "smartphone", "mobile phone"],
        "microwave": ["microwave", "microwave oven"],
        "oven": ["oven", "stove", "cooker"],
        "toaster": ["toaster", "appliance"],
        "refrigerator": ["refrigerator", "fridge", "freezer"],
        "book": ["book", "novel", "textbook"],
        "clock": ["clock", "timepiece"],
        "vase": ["vase", "flower vase"],
        "scissors": ["scissors", "shears"],
        "teddy bear": ["teddy bear", "stuffed animal", "toy"],
        "hair drier": ["hair drier", "hair dryer"],
        "toothbrush": ["toothbrush", "brush"]
    }

    # 遥感领域专用词库
    REMOTE_SENSING_MAPPINGS: Dict[str, List[str]] = {
        # 建筑物
        "building": ["building", "house", "structure", "house", "residential building", "commercial building",
                     "apartment", "dwelling", "construction", "edifice"],
        "house": ["house", "home", "residence", "dwelling", "bungalow"],
        "factory": ["factory", "manufacturing plant", "industrial building", "warehouse", "mill"],
        "school": ["school", "academy", "educational building", "classroom building"],
        "hospital": ["hospital", "medical center", "clinic", "healthcare facility"],
        "stadium": ["stadium", "sports arena", "arena", "sports field"],
        "tower": ["tower", "observation tower", "communication tower", "antenna"],

        # 交通
        "road": ["road", "highway", "street", "lane", "avenue", "pathway", "motorway"],
        "bridge": ["bridge", "overpass", "viaduct", "flyover"],
        "parking lot": ["parking lot", "parking area", "car park", "parking space"],
        "airport": ["airport", "airfield", "runway", "heliport", "aviation facility"],
        "railway": ["railway", "railroad", "train tracks", "rail line"],
        "port": ["port", "harbor", "dock", "seaport", "marina"],

        # 车辆（遥感场景）
        "parked car": ["parked car", "parked vehicle", "parked automobile", "parked cars"],
        "moving vehicle": ["moving vehicle", "moving car", "traffic", "vehicles in motion"],
        "ship": ["ship", "vessel", "boat", "cargo ship", "container ship", "tanker"],
        "fishing boat": ["fishing boat", "fishing vessel", "fishing trawler"],

        # 自然地物
        "river": ["river", "stream", "waterway", "creek", "brook"],
        "lake": ["lake", "pond", "reservoir", "lagoon"],
        "sea": ["sea", "ocean", "water body", "coastal water"],
        "forest": ["forest", "woods", "trees", "woodland", "tree cover", "vegetation"],
        "tree": ["tree", "trees", "plant", "vegetation", "individual tree"],
        "grass": ["grass", "lawn", "grassland", "meadow", "green area"],
        "farmland": ["farmland", "cropland", "agricultural land", "farm", "cultivated land", "field"],
        "field": ["field", "farmland", "crop field", "agricultural field", "cultivated field"],
        "beach": ["beach", "shore", "shoreline", "coastal area"],
        "desert": ["desert", "arid land", "sandy area", "barren land"],
        "mountain": ["mountain", "hill", "peak", "summit", "hillside"],
        "cloud": ["cloud", "clouds", "cloud cover", "cloudy region"],

        # 其他
        "solar panel": ["solar panel", "photovoltaic panel", "solar array", "solar farm"],
        "swimming pool": ["swimming pool", "pool", "water pool"],
        "container": ["container", "shipping container", "cargo container", "集装箱"],
        "oil tank": ["oil tank", "storage tank", "petroleum tank", "fuel tank"]
    }

    def __init__(self, enable_remote_sensing: bool = True):
        """
        初始化提示词增强器

        Args:
            enable_remote_sensing: 是否启用遥感词库
        """
        self._enable_remote_sensing = enable_remote_sensing
        self._build_combined_mappings()

    def _build_combined_mappings(self):
        """合并词库"""
        self._mappings = {}

        # 添加通用词库
        for key, synonyms in self.GENERAL_MAPPINGS.items():
            self._mappings[key.lower()] = synonyms

        # 添加遥感词库
        if self._enable_remote_sensing:
            for key, synonyms in self.REMOTE_SENSING_MAPPINGS.items():
                if key.lower() in self._mappings:
                    # 合并同义词
                    existing = set(self._mappings[key.lower()])
                    existing.update(synonyms)
                    self._mappings[key.lower()] = list(existing)
                else:
                    self._mappings[key.lower()] = synonyms

    def set_remote_sensing(self, enable: bool):
        """设置是否启用遥感词库"""
        self._enable_remote_sensing = enable
        self._build_combined_mappings()

    def enhance(self, prompt: str) -> str:
        """
        增强提示词

        Args:
            prompt: 原始提示词，如 "car, person, building"

        Returns:
            增强后的提示词，如 "car vehicle automobile sedan suv truck van, person people human pedestrian"
        """
        # 解析原始提示词
        categories = self._parse_prompt(prompt)

        # 扩展每个类别
        enhanced_categories = []
        for category in categories:
            enhanced = self._expand_category(category)
            enhanced_categories.append(", ".join(enhanced))

        return ", ".join(enhanced_categories)

    def enhance_list(self, prompt: str) -> List[str]:
        """
        增强提示词并返回词列表

        Args:
            prompt: 原始提示词

        Returns:
            增强后的词列表
        """
        categories = self._parse_prompt(prompt)
        result = []

        for category in categories:
            result.extend(self._expand_category(category))

        return result

    def _parse_prompt(self, prompt: str) -> List[str]:
        """解析提示词，分割成类别列表"""
        import re
        # 按逗号、分号、空格分割
        parts = re.split(r'[,\s;]+', prompt)
        # 清理并过滤空字符串
        categories = [p.strip().lower() for p in parts if p.strip()]
        return categories

    def _expand_category(self, category: str) -> List[str]:
        """
        扩展单个类别

        Args:
            category: 类别名称

        Returns:
            扩展后的同义词列表
        """
        # 直接查找
        if category in self._mappings:
            return self._mappings[category]

        # 部分匹配（category 是某个词条的子串）
        for key, synonyms in self._mappings.items():
            if category in key or key in category:
                return synonyms

        # 没有找到匹配，返回原始类别
        return [category]

    def get_all_categories(self) -> List[str]:
        """获取所有可用的类别名称"""
        return list(self._mappings.keys())

    def get_suggestions(self, partial: str) -> List[str]:
        """
        获取部分输入的建议

        Args:
            partial: 部分输入

        Returns:
            建议列表
        """
        partial = partial.lower().strip()
        if not partial:
            return self.get_all_categories()

        suggestions = []
        for category in self._mappings.keys():
            if category.startswith(partial):
                suggestions.append(category)

        return sorted(suggestions)
