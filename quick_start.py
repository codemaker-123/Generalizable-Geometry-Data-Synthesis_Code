import sys
import os
from utils.loading_utils import load_definitions_and_rules
import graph as gh
import problem as pr
from clause_generation import * #CompoundClauseGen
# import signal
import threading
from pretty_problem_statement_dict import * # 改为经过语料扩展的结果
import json
# from data_augmentation.opencv import *
import shutil
import geometry as gm
sys.path.append('..')
# signal.signal(signal.SIGALRM, signal_handler)
data_desc=[]

save_dir= './'
defs_path = './defs.txt'
rules_path = './rules.txt'
complexity = 2# set complexity
definitions, rules = load_definitions_and_rules(defs_path, rules_path)
# i = 70000
cc_gen = CompoundClauseGen(definitions, complexity)

# Automatic clause select
txt = cc_gen.generate_clauses()  
p = pr.Problem.from_txt(txt)
print(txt)

def timeout_handler():
    raise TimeoutException("Time's up!")

angles = []
equ_angle = []
for cl in p.clauses:
    for cons in cl.constructions:
        if "s_angle" in str(cons):  # 直接转字符串匹配
            print("找到角度构造:", cons)
            # 按照构造角的顺序，顶点为 cons.vertex，其它两点为 cons.p1 和 cons.p2
            # 注：这里可以根据实际需求调整顺序
            p1 = cons.args[0]
            head = cons.args[1]
            p2 = cons.args[2]
            angle = cons.args[3]
            # 保存时可以附带预定义角度数值，后续可用于验证或标注
            angles.append((p1, head, p2, angle))
        # if "angle_mirror" in str(cons):
        #     print("找到angle_mirror:", cons)
        #     p0 = cons.args[0]
        #     p1 = cons.args[1]
        #     p2 = cons.args[2]
        #     p3 = cons.args[3] #123 320
        #     equ_angle.append([(p1,p2,p3),(p3,p2,p0)])
        #     print(equ_angle)
        # if "angle_bisector" in str(cons):
        #     print("找到angle_mirror:", cons)
        #     p0 = cons.args[0]
        #     p1 = cons.args[1]
        #     p2 = cons.args[2]
        #     p3 = cons.args[3] #123 320
        #     equ_angle.append([(p1,p2,p0),(p0,p2,p3)])
        #     print(equ_angle)
        # if "eqangle3" in str(cons):
        #     print("找到eqangle3:", cons)
        #     p0 = cons.args[0]
        #     p1 = cons.args[1]
        #     p2 = cons.args[2]
        #     p3 = cons.args[3] #123 320
        #     p4 = cons.args[4]
        #     p5 = cons.args[5]
        #     equ_angle.append([(p1,p0,p2),(p4,p3,p5)])
        #     print(equ_angle)
try:


#     ]
    # angle_equal = [,"angle_bisector",'angle_mirror', 'eqangle3' ]
    g, _ = gh.Graph.build_problem(p, definitions)
    para = []  # 用于存储符合条件的子列表

    # 遍历原始列表，检查每个子列表
    for sublist in list(g.cache.keys()):
        if 'para' in sublist:
            para.append(sublist)
    equ_angle = [[n.name for n in segment.equivs()] for segment in g.type2nodes[gh.Measure]]
    len_name_len  =  gh.nm.draw_reinforce(
        g.type2nodes[gh.Point],
        g.type2nodes[gh.Line],
        g.type2nodes[gh.Circle],
        g.type2nodes[gh.Length],
        theme='',
        angle = angles,
        equ_angle = equ_angle,
        para = para,
        save_to=save_dir+f"/demo.jpg")
    print(len_name_len)
    # print([[n.name for n in segment.equivs()] for segment in g.type2nodes[gh.Measure]])
    # 过滤掉 None 数据
    filtered_data = [(length, name) for length, name in len_name_len if length is not None and name is not None]

    # 生成句子
    # Assuming filtered_data is a list of tuples (length, name)
# Assuming filtered_data is a list of tuples (length, name)
# Assuming filtered_data is a list of tuples (length, name)
# Assuming filtered_data is a list of tuples (length, name)
    if filtered_data:
        # Generate the main sentence parts
        sentences = [f"the length of {name} is {length:.2f}" for length, name in filtered_data]
        
        # Capitalize only the first sentence (without altering the name capitalization)
        sentences[0] = sentences[0][0].upper() + sentences[0][1:]

        # Join all but the last element with commas and then append 'and' before the last one
        if len(sentences) > 1:
            result = ", ".join(sentences[:-1]) + " and " + sentences[-1]
        else:
            result = sentences[0]
        result = result + "."
    else:
        result = ""

    print(result)


    data_desc.append(
        {
        "id": f"demo",
        "image": f"img/demo.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "Render a clear and concise description of a image about geometric shapes.\n<image>"
            },
            {
                "from": "gpt",
                "value": gen_nl(txt) + result 
            }
        ],
        "clause": [remove_uppercase_space(clause_item) for clause_item in txt.split(";")]
    }
    )
        
#     signal.alarm(0)
# except KeyboardInterrupt:
#     sys.exit(0)
# except:
#     print('err occurred, retrying ...')
#     sys.exit(0)
    # t.cancel()  # 如果代码提前执行完毕，取消定时器
except TimeoutException as e:
    print(e)
json_data = json.dumps(data_desc, indent=2)
with open(f"{save_dir}/data.json", "w") as file:
    file.write(json_data)
