#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 22:15:56 2019

@author: yaoxinzhi
"""

'''
sentence	Mineralogical and geochemical analyses showed two main types of intimately associated products: a polymetallic sulfide-rich material composed of pyrite and marcasite in association, zinc-rich phases, and copper-rich compounds, and an iron-rich oxide and hydroxide material (also called gossan) composed largely of goethite and limonite.
annotation	327|335|limonite|Chemical|MESH:C021024
annotation	254|263|hydroxide|Chemical|MESH:C031356
annotation	204|210|copper|Chemical|MESH:D003300
annotation	145|151|pyrite|Chemical|MESH:C011342
'''



import copy
from stanfordcorenlp import StanfordCoreNLP

# 定义树结点类 储存 节点值 与 子节点 列表
class Node:
    def __init__(self, val):
        self.value = val
        self.child_list = []
        
    def add_child(self, node):
        self.child_list.append(node)
        
# 深度优先查找 返回跟节点到目标节点的路径 只查子孩子
def deep_first_search(cur, val, path=[]):
    # 当前节点值添加路径列表
    path.append(cur.value)
#    print (path)
    # 如果找到目标 返回路径列表
    if cur.value == val:
        return path
    # 如果没有孩子列表 就返回 no 回溯标记
    if cur.child_list == []:
        return 'no'
    # 对孩子列表中的每个孩子进行递归
    for node in cur.child_list:
        # 深拷贝当前路径列表
        t_path = copy.deepcopy(path)
#        t_path = path
        res = deep_first_search(node, val, t_path)
        # 如果返回no 说明找到了叶子节点没有找到 利用临时路径继续找下一个孩子节点
        if res == 'no':
            continue
        else:
            # 如果返回的不是 no 说明找到了路径
            return res
    # 如果所有孩子都没找到 则回溯
    return 'no'

# 深度优先查找 非递归写法
def deep_first_search_non_recursive(root, val):
    # sentence 太长造成迭代深度优先遍历 迭代深度爆炸 所以牺牲内存写个非迭代
    path = []
    # 如果要找的是根节点 直接返回结果
    if root.value == val:
        path.append(val)
        return path
    # 栈
    stack_list = []
    # 被访问标记
    visited = []
    # 栈内加入已遍历节点
    stack_list.append(root)
    # path 变量记录 路径
    path.append(root.value)
    # 访问过的节点 添加进 visited 列表
    visited.append(root)
    # 深度优先遍历整棵树
    while len(stack_list) > 0:
        x = stack_list[-1]
        for w in x.child_list:
            # 遍历节点所有未被访问的子孩子
            if not w in visited:
                # 如果找到 返回path 结束遍历
                if w.value == val:
                    path.append(val)
                    return path
                
                visited.append(w)
                stack_list.append(w)
                path.append(w.value)           
                break
        # 在遍历当前 x 节点后依旧没有找到目标节点 -- x出栈
        if stack_list[-1] == x:
            stack_list.pop()
            path.pop()


def get_shortest_path(root, start, end):
    # 基本思想： 多叉树两个结点间距离就是两个结点共同父结点到两个节点的距离和
    # 分别获取从根节点到 start 和 end 的路径列表 如果没有目标节点 就返回no
    path_start = deep_first_search_non_recursive(root, start)
    path_end = deep_first_search_non_recursive(root, end)
    # 对两个路径 从尾巴开始向头 找到最近的公共根节点 合并根节点
    if not path_start or not path_end:
        return None
    len_start = len(path_start)
    for i in range(len_start-1, -1, -1):
        if path_start[i] in path_end:
            index = path_end.index(path_start[i])
            path_end = path_end[index:]
            path_start = path_start[-1: i: -1]
            break
    res = path_start + path_end
    distace = len(res) - 1
    path = '->'.join(res)
    return '{0}:{1}'.format(distace, path)


def init_tree_StanfordCoreNLP(dependency_result, tokens):
    # 使用StanfordCoreNLP 处理 保证 依存树结点同分词结果相同  
    # 依存句法分析
    
    # 找根节点（'ROOT', 0, index) 并创建根节点
    _root = [i for i in dependency_result if i[0] == 'ROOT'][0][2]
    root = Node(str(_root))
#    print ('successful constraction the root -> name: {0}'.format(_root))
    # 把每个节点都给构建了吧
    for i in range(len(tokens)):
        node_name = '_{0}'.format(i+1)
        exec('{0} = Node("{1}")'.format(node_name, i+1))
#        print ('successful constraction the node -> name: {0}, value: {1}'.format(node_name, i+1))
    # 根据 依存分析 结果 构建节点 父子关系
    for i in dependency_result:
        if i[0] != 'ROOT':
            child_name = '_{0}'.format(i[2])
            if str(i[1]) == str(_root):
                exec('root.add_child({0})'.format(child_name))
            else:
                parent_name = '_{0}'.format(i[1])
                exec('{0}.add_child({1})'.format(parent_name, child_name))
    return root

def distence_pro(distance, token_index, dep_res):
    # 添加保存依存路径类型
    dep_dic = {}
    for dep_pair in dep_res:
        dep_dic[(dep_pair[1], dep_pair[2])] = dep_pair[0]
    
    _path = []
    path_result = []
    # 该函数用于 distance 中 依存路径 转化为 word
    dis = distance.split(':')[0]
#    print (distance)
    path = [str(i) for i in distance.split(':')[1].split('->')]
#    _path = [token_index[j] for j in path]
    for i in range(1, len(path)):
        _path.append(path[i-1])
        for j in dep_res:
            if (str(j[1]) == path[i-1] and str(j[2]) == path[i]) or (str(j[1]) == path[i] and str(j[2]) == path[i-1]):
                _path.append(j[0])
    _path.append(path[-1])
    
    for i in _path:
        
        if token_index.get(i):
            path_result.append(token_index[i])
        else:
            path_result.append(i)
    return '{0}:{1}'.format(dis, '->'.join(path_result))

def main():
    nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05')
    
    # 选择一个句子
    sentence = 'Mineralogical and geochemical analyses showed two main types of intimately associated products: a polymetallic sulfide-rich material composed of pyrite and marcasite in association, zinc-rich phases, and copper rich compounds, and an iron-rich oxide and hydroxide material (also called gossan) composed largely of goethite and limonite.'
    
    # 依存句法分析 举个栗子 看看解析结果格式
    dependency_result = nlp.dependency_parse(sentence)
    print ('DependencyParsingResult:\n{0}\n'.format(dependency_result))
    
    # 对句子分词 并且标注每个词位置 防止有一样的词
    # StanfordCoreNLP 流程化处理 保证分词和树节点一致
    tokens = nlp.word_tokenize(sentence)
        
    # 为了查找记录每个单词的index
    token_index_dic = {}
    for index, value in enumerate(tokens):
        token_index_dic[str(index+1)] = value
        
    print ('token_index_dic:\n{0}\n'.format(token_index_dic))
    
    # 构建树
    root = init_tree_StanfordCoreNLP(dependency_result, tokens)
    
    # '50': 'limonite' '38': 'hydroxide' 
    # '30': 'copper' '20': 'pyrite'
    start_path = deep_first_search_non_recursive(root, '30')
    print ("path from root to copper :\n{0} ".format(start_path))
    end_path = deep_first_search_non_recursive(root, '20')
    print ("path from root to pyrite :\n{0}\n ".format(end_path))
    
    path = get_shortest_path(root, '30', '20')
    print ('path between "copper" and "pyrite":\n{0}\n'.format(path))
    
    SDP = distence_pro(path, token_index_dic, dependency_result)
    print ('SDP between "copper" and "pyrite":\n{0}'.format(SDP))
    
    # Do not forget to close! The backend server will consume a lot memery
    nlp.close()
    
if __name__ == '__main__':
    main()
