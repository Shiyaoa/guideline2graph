"""
医学术语词典配置

包含：
- 医学缩写扩展
- 中文→英文映射
- 药物类别映射
- 特殊概念映射
"""

# ==================== 医学缩写扩展 ====================

ABBREVIATION_DICT = {
    # 药物类
    "SGLT2i": "Sodium glucose cotransporter-2 inhibitor",
    "SGLT2": "Sodium glucose cotransporter-2",
    "ACEI": "Angiotensin converting enzyme inhibitor",
    "ACEI": "Angiotensin-converting enzyme inhibitor",
    "ARB": "Angiotensin receptor blocker",
    "RASI": "Renin angiotensin system inhibitor",
    "MRA": "Mineralocorticoid receptor antagonist",
    "BB": "Beta blocker",
    "DPP4i": "Dipeptidyl peptidase-4 inhibitor",
    "DPP-4i": "Dipeptidyl peptidase-4 inhibitor",
    "GLP1RA": "Glucagon-like peptide-1 receptor agonist",
    "GLP-1RA": "Glucagon-like peptide-1 receptor agonist",
    "TZD": "Thiazolidinedione",
    "SU": "Sulfonylurea",

    # 疾病类
    "HFrEF": "Heart failure with reduced ejection fraction",
    "HFpEF": "Heart failure with preserved ejection fraction",
    "HFmrEF": "Heart failure with mildly reduced ejection fraction",
    "T2DM": "Type 2 diabetes mellitus",
    "T1DM": "Type 1 diabetes mellitus",
    "DM": "Diabetes mellitus",
    "CKD": "Chronic kidney disease",
    "CVD": "Cardiovascular disease",
    "CAD": "Coronary artery disease",
    "CHD": "Coronary heart disease",
    "MI": "Myocardial infarction",
    "AF": "Atrial fibrillation",
    "AFL": "Atrial flutter",
    "HF": "Heart failure",
    "ESRD": "End stage renal disease",
    "AKI": "Acute kidney injury",
    "DKD": "Diabetic kidney disease",

    # 检查/测量类
    "eGFR": "Estimated glomerular filtration rate",
    "GFR": "Glomerular filtration rate",
    "HbA1c": "Hemoglobin A1c",
    "A1c": "Hemoglobin A1c",
    "BNP": "Brain natriuretic peptide",
    "NT-proBNP": "N-terminal pro-brain natriuretic peptide",
    "SBP": "Systolic blood pressure",
    "DBP": "Diastolic blood pressure",
    "BMI": "Body mass index",
    "LDL-C": "Low density lipoprotein cholesterol",
    "HDL-C": "High density lipoprotein cholesterol",
    "TG": "Triglycerides",
    "TC": "Total cholesterol",
    "ALT": "Alanine aminotransferase",
    "AST": "Aspartate aminotransferase",
    "Cr": "Creatinine",
    "K": "Potassium",
}

# ==================== 中文到英文医学术语映射 ====================

CHINESE_ENGLISH_DICT = {
    # 药物 - SGLT2抑制剂
    "达格列净": "dapagliflozin",
    "恩格列净": "empagliflozin",
    "卡格列净": "canagliflozin",
    "厄托格列净": "ertugliflozin",
    "索格列净": "sotagliflozin",

    # 药物 - 其他降糖药
    "二甲双胍": "metformin",
    "胰岛素": "insulin",
    "格列美脲": "glimepiride",
    "格列齐特": "gliclazide",
    "格列本脲": "glyburide",
    "利拉鲁肽": "liraglutide",
    "司美格鲁肽": "semaglutide",
    "艾塞那肽": "exenatide",
    "度拉糖肽": "dulaglutide",
    "西格列汀": "sitagliptin",
    "利格列汀": "linagliptin",
    "沙格列汀": "saxagliptin",
    "阿卡波糖": "acarbose",
    "吡格列酮": "pioglitazone",

    # 药物 - 心血管
    "利尿剂": "diuretic",
    "袢利尿剂": "loop diuretic",
    "呋塞米": "furosemide",
    "托拉塞米": "torsemide",
    "布美他尼": "bumetanide",
    "螺内酯": "spironolactone",
    "依普利酮": "eplerenone",
    "β受体阻滞剂": "beta blocker",
    "β阻滞剂": "beta blocker",
    "倍他阻滞剂": "beta blocker",
    "美托洛尔": "metoprolol",
    "比索洛尔": "bisoprolol",
    "卡维地洛": "carvedilol",
    "阿替洛尔": "atenolol",
    "普萘洛尔": "propranolol",
    "ACEI": "angiotensin converting enzyme inhibitor",
    "血管紧张素转换酶抑制剂": "angiotensin converting enzyme inhibitor",
    "依那普利": "enalapril",
    "赖诺普利": "lisinopril",
    "培哚普利": "perindopril",
    "雷米普利": "ramipril",
    "卡托普利": "captopril",
    "ARB": "angiotensin receptor blocker",
    "血管紧张素受体拮抗剂": "angiotensin receptor blocker",
    "氯沙坦": "losartan",
    "缬沙坦": "valsartan",
    "厄贝沙坦": "irbesartan",
    "坎地沙坦": "candesartan",
    "替米沙坦": "telmisartan",
    "洋地黄": "digitalis",
    "地高辛": "digoxin",
    "阿司匹林": "aspirin",
    "华法林": "warfarin",
    "达比加群": "dabigatran",
    "利伐沙班": "rivaroxaban",
    "阿哌沙班": "apixaban",
    "他汀": "statin",
    "阿托伐他汀": "atorvastatin",
    "瑞舒伐他汀": "rosuvastatin",
    "辛伐他汀": "simvastatin",

    # 疾病 - 糖尿病相关
    "糖尿病": "diabetes mellitus",
    "1型糖尿病": "type 1 diabetes mellitus",
    "2型糖尿病": "type 2 diabetes mellitus",
    "糖尿病肾病": "diabetic nephropathy",
    "糖尿病视网膜病变": "diabetic retinopathy",
    "糖尿病神经病变": "diabetic neuropathy",
    "糖尿病足": "diabetic foot",
    "低血糖": "hypoglycemia",
    "高血糖": "hyperglycemia",
    "酮症酸中毒": "ketoacidosis",

    # 疾病 - 心血管
    "心衰": "heart failure",
    "心力衰竭": "heart failure",
    "射血分数降低的心衰": "heart failure with reduced ejection fraction",
    "射血分数保留的心衰": "heart failure with preserved ejection fraction",
    "心梗": "myocardial infarction",
    "心肌梗死": "myocardial infarction",
    "冠心病": "coronary heart disease",
    "冠状动脉粥样硬化性心脏病": "coronary artery disease",
    "心绞痛": "angina pectoris",
    "房颤": "atrial fibrillation",
    "心房颤动": "atrial fibrillation",
    "房扑": "atrial flutter",
    "心血管疾病": "cardiovascular disease",
    "高血压": "hypertension",
    "低血压": "hypotension",
    "动脉粥样硬化": "atherosclerosis",

    # 疾病 - 肾脏
    "肾病": "kidney disease",
    "慢性肾病": "chronic kidney disease",
    "慢性肾脏病": "chronic kidney disease",
    "终末期肾病": "end stage renal disease",
    "肾衰竭": "renal failure",
    "急性肾损伤": "acute kidney injury",
    "蛋白尿": "proteinuria",
    "微量白蛋白尿": "microalbuminuria",

    # 检查/测量
    "血糖": "blood glucose",
    "空腹血糖": "fasting blood glucose",
    "餐后血糖": "postprandial blood glucose",
    "随机血糖": "random blood glucose",
    "糖化血红蛋白": "hemoglobin A1c",
    "血压": "blood pressure",
    "收缩压": "systolic blood pressure",
    "舒张压": "diastolic blood pressure",
    "肾小球滤过率": "glomerular filtration rate",
    "体重": "body weight",
    "体重指数": "body mass index",
    "尿蛋白": "urine protein",
    "血肌酐": "serum creatinine",
    "血脂": "blood lipid",

    # 观察/状态
    "妊娠": "pregnancy",
    "孕妇": "pregnant woman",
    "哺乳期": "lactating",
    "哺乳": "breastfeeding",
    "不良反应": "adverse reaction",
    "药物过敏": "drug allergy",
    "住院": "hospitalization",
    "心衰住院": "heart failure hospitalization",
    "死亡": "death",
    "心血管死亡": "cardiovascular death",
    "全因死亡": "all-cause mortality",
}

# ==================== 药物类别到具体药物映射 ====================

DRUG_CLASS_MEMBERS = {
    "SGLT2 inhibitor": ["dapagliflozin", "empagliflozin", "canagliflozin", "ertugliflozin"],
    "SGLT2i": ["dapagliflozin", "empagliflozin", "canagliflozin", "ertugliflozin"],
    "beta blocker": ["metoprolol", "carvedilol", "bisoprolol", "atenolol", "propranolol"],
    "beta-blocker": ["metoprolol", "carvedilol", "bisoprolol", "atenolol", "propranolol"],
    "diuretic": ["furosemide", "hydrochlorothiazide", "spironolactone", "torsemide"],
    "loop diuretic": ["furosemide", "torsemide", "bumetanide"],
    "ACE inhibitor": ["lisinopril", "enalapril", "ramipril", "captopril", "perindopril"],
    "ACEI": ["lisinopril", "enalapril", "ramipril", "captopril", "perindopril"],
    "ARB": ["losartan", "valsartan", "irbesartan", "candesartan", "telmisartan"],
    "MRA": ["spironolactone", "eplerenone"],
    "statin": ["atorvastatin", "rosuvastatin", "simvastatin", "pravastatin"],
    "insulin": ["insulin glargine", "insulin detemir", "insulin lispro", "insulin aspart"],
    "GLP-1 agonist": ["liraglutide", "semaglutide", "exenatide", "dulaglutide"],
    "DPP-4 inhibitor": ["sitagliptin", "linagliptin", "saxagliptin", "alogliptin"],
    "metformin": ["metformin"],
    "sulfonylurea": ["glimepiride", "gliclazide", "glyburide", "glipizide"],
}

# ==================== 特殊概念映射 ====================
# 格式: (term_lower, domain_id) -> concept_id
# 用于已知OMOP中存在但名称匹配困难的术语

SPECIAL_MAPPINGS = {
    # SGLT2抑制剂 - ATC药物类别
    ("sglt2i", "Drug"): 1123627,
    ("sglt2抑制剂", "Drug"): 1123627,
    ("钠-葡萄糖协同转运蛋白2抑制剂", "Drug"): 1123627,
    ("sglt2 inhibitor", "Drug"): 1123627,
    ("sglt2 inhibitors", "Drug"): 1123627,  # 复数形式
    ("sodium-glucose cotransporter-2 inhibitor", "Drug"): 1123627,
    ("sodium glucose cotransporter 2 inhibitor", "Drug"): 1123627,
    ("sodium-glucose co-transporter 2 inhibitors", "Drug"): 1123627,
    ("sodium glucose co-transporter 2 inhibitor", "Drug"): 1123627,

    # 心血管药物类 - ATC 药物类别概念
    ("利尿剂", "Drug"): 21601461,
    ("diuretic", "Drug"): 21601461,
    ("diuretics", "Drug"): 21601461,  # ATC 2nd: DIURETICS
    ("β受体阻滞剂", "Drug"): 21603698,
    ("β阻滞剂", "Drug"): 21603698,
    ("beta blocker", "Drug"): 21603698,
    ("beta blockers", "Drug"): 21603698,  # ATC 4th: Beta blocking agents
    ("beta-blocker", "Drug"): 21603698,
    ("beta-blockers", "Drug"): 21603698,
    ("rasi", "Drug"): 1308216,
    ("ace inhibitor", "Drug"): 1308216,
    ("ace inhibitors", "Drug"): 1308216,
    ("acei", "Drug"): 1308216,
    ("arb", "Drug"): 1367500,
    ("arbs", "Drug"): 1367500,
    ("mra", "Drug"): 970250,
    ("mras", "Drug"): 970250,
    ("mineralocorticoid receptor antagonist", "Drug"): 970250,
    ("mineralocorticoid receptor antagonists", "Drug"): 970250,

    # 降糖药物类 - ATC药物类别概念
    ("dpp-4 inhibitor", "Drug"): 21600783,
    ("dpp-4 inhibitors", "Drug"): 21600783,  # 复数形式
    ("dpp4i", "Drug"): 21600783,
    ("dipeptidyl peptidase-4 inhibitor", "Drug"): 21600783,
    ("glp-1 receptor agonist", "Drug"): 1123618,
    ("glp-1 receptor agonists", "Drug"): 1123618,  # 复数形式
    ("glp-1 analogue", "Drug"): 1123618,
    ("glp-1 analogues", "Drug"): 1123618,  # 复数形式
    ("glp1ra", "Drug"): 1123618,
    ("sulfonylurea", "Drug"): 21600749,
    ("sulfonylureas", "Drug"): 21600749,  # 复数形式
    ("thiazolidinedione", "Drug"): 21600779,
    ("thiazolidinediones", "Drug"): 21600779,  # 复数形式
    ("tzd", "Drug"): 21600779,
    ("biguanide", "Drug"): 21600745,
    ("biguanides", "Drug"): 21600745,  # 复数形式
    ("insulin", "Drug"): 21600713,
    ("insulins", "Drug"): 21600713,  # 复数形式

    # 疾病类
    ("hfref", "Condition"): 45766164,
    ("hfpef", "Condition"): 40486933,
    ("低血压", "Condition"): 316447,
    ("hypotension", "Condition"): 316447,

    # 测量类
    ("egfr", "Measurement"): 3655421,
    ("estimated glomerular filtration rate", "Measurement"): 3655421,
    ("肾小球滤过率", "Measurement"): 3655421,
}

# ==================== 领域映射 ====================
# 将中文实体类型映射到OMOP Domain

DOMAIN_MAPPING = {
    "Condition": "Condition",
    "疾病": "Condition",
    "诊断": "Condition",
    "症状": "Condition",
    "Drug": "Drug",
    "药物": "Drug",
    "药品": "Drug",
    "Procedure": "Procedure",
    "操作": "Procedure",
    "手术": "Procedure",
    "检查": "Procedure",
    "Measurement": "Measurement",
    "测量": "Measurement",
    "指标": "Measurement",
    "实验室检查": "Measurement",
    "Observation": "Observation",
    "观察": "Observation",
    "评估": "Observation",
    "状态": "Observation",
}