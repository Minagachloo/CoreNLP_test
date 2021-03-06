#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 22:58:02 2019

@author: yaoxinzhi
"""

'''
该代码使用 StanfordCoreNLP 实验测试
'''

import os
import copy
import time
import argparse
from itertools import combinations
from stanfordcorenlp import StanfordCoreNLP

# 定义树结点类 储存 节点值 与 子节点 列表
class Node:
    def __init__(self, val):
        self.value = val
        self.child_list = []
        
    def add_child(self, node):
        self.child_list.append(node)
        
# 深度查找优先 返回根节点到目标节点的路径 (递归写法)
def deep_first_search(cur, val, path=[]):
    # 当前节点值添加路径列表
    path.append(cur.value)
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


def init_tree_StanfordCoreNLP(sent, tokens):
    # 使用StanfordCoreNLP 处理 并储存为树结构 保证 依存树结点同分词结果相同
    # 依存句法分析
    dependency_result = nlp.dependency_parse(sent)
    print('Dependency paring result:\n{0}'.format(dependency_result))
    # 找根节点（'ROOT', 0, index) 并创建根节点
    _root = [i for i in dependency_result if i[0] == 'ROOT'][0][2]
    root = Node(str(_root))
    # 构建每个树节点
    for i in range(len(tokens)):
        node_name = '_{0}'.format(i+1)
        exec('{0} = Node("{1}")'.format(node_name, i+1))   
    # 根据 依存分析 结果 构建节点 父子关系
    for i in dependency_result:
        if i[0] != 'ROOT':
            child_name = '_{0}'.format(i[2])
            # 根节点的名字是 root 
            if str(i[1]) == str(_root):
                exec('root.add_child({0})'.format(child_name))
            else:
                parent_name = '_{0}'.format(i[1])
                exec('{0}.add_child({1})'.format(parent_name, child_name))
    return root, dependency_result
            
def sub_sent(string, start, end, old=' ', new='_'):
    # 实体中出现多个空格 替换为一个 '-'
    _string = string[:start] +  ' '.join(string[start:end].split()).replace(old, new)+ string[end:]
    return _string

def sub_word(word, old=' ', new='_'):
    # 替换词组中的空格
    return ' '.join(word.split()).replace(old, new)

def get_offset(tokens, sent):
    # 该函数用于或取每个 token 的 offset 信息
    start = 0
    _len = 0
    token_offset_dic = {}
    for index, token in enumerate(tokens):
        start = start + sent[start:].find(token)
        offset = (str(start), str(start + len(token)))
        token_offset_dic[offset] = (token, index + 1)
        start += len(token)
        _len += len(token)
    return token_offset_dic

def phrase_pro(sent, ann_list):
    # 该函数用于解决 pubtator 注释中存在词组的问题
    _sent = sent
    _ann_list = []
    for index, value in enumerate(ann_list):
        if ' ' in value[1]: 
            _ann_list.append((value[0], sub_word(value[1]), value[2], value[3]))
            _start = int(value[0][0])
            _end = int(value[0][1])
            _sent = sub_sent(_sent, _start, _end)
        else:
            _ann_list.append(value)
    return _ann_list, _sent
            

def pre_processing(ann_list, token_offset):
    # 该函数用于解决 sentences 中 pubtator 分词和依存树节点不匹配问题 只注释了分词的部分
    _ann_list = []
    for ann in ann_list:
        span = ann[0]
        if token_offset.get(span):
            _ann_list.append((span, ann[1], ann[2], ann[3], token_offset[span][1]))
        else:
            _start = int(span[0])
            _end = int(span[1])
            for k in token_offset.keys():
                if (_start > int(k[0]) or _start == int(k[0])) and (_end < int(k[1]) or _end == int(k[1])):
                    _ann_list.append(((k[0], k[1]), token_offset[k][0], ann[2], ann[3], token_offset[k][1]))
    return _ann_list

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

def DependencyParsing(rf, outfile):
    
    wf = open(outfile, 'w')
    count = 0
    ann_list = []
    total_cost = 0
    sent = ''
    
    with open(rf) as f:
        
        # 文件读取
        for line in f:
            l = line.replace('\n', '').split('\t')
            if l != ['']:
                time_start = time.time()
                if l[0] == 'sentence':
                    sent = l[1]
                    count += 1

                if l[0] == 'annotation':
                    ann = l[1].split('|')
                    span = (ann[0], ann[1])
                    entity = (span, ann[2], ann[3], ann[4])
                    if entity not in ann_list:
                        ann_list.append(entity)
            # 到空行处理数据
            if l == [''] and sent != '':
                # 解决pubtaotr注释中存在空格 依存树分为了两个节点的问题
                ann_list, sent = phrase_pro(sent, ann_list)
                
                # 分词并或取 tokens 的 offset 及 index 信息
                tokens = nlp.word_tokenize(sent)
                token_offset = get_offset(tokens, sent)
                token_index_dic = {}
                for index, value in enumerate(tokens):
                    token_index_dic[str(index+1)] = value
                # 注释 和 句子 预处理 
                ann_list = pre_processing(ann_list, token_offset)
                
                # 自由组合 及 计算依存距离
                _tree, dep_res = init_tree_StanfordCoreNLP(sent, tokens)
                entity_pair = list(combinations(ann_list, 2))
                wf.write('sentence\t{0}\n'.format(sent))
                for pair in entity_pair:
                    en1, en2 = pair[0], pair[1]
                    distance = get_shortest_path(_tree, str(en1[4]), str(en2[4]))
                    if distance:
                        distance = distence_pro(distance, token_index_dic, dep_res)
                    else:
                        distance = 'None'
                    en1_w = '{0}|{1}|{2}|{3}|{4}'.format(en1[0][0], en1[0][1], en1[1], en1[2], en1[3])
                    en2_w = '{0}|{1}|{2}|{3}|{4}'.format(en2[0][0], en2[0][1], en2[1], en2[2], en2[3])
                    wf.write('{0}\t{1}\t{2}\n'.format(distance, en1_w, en2_w))
       

                time_end = time.time()
                cost = time_end - time_start
                total_cost += cost         
                if count % 300 == 0:                
                    print ('sentences: {0} cost: {1}'.format(count, total_cost))
                
#               将变量归空用来判断连续的空行
                wf.write('\n')
                sent = ''
                span = ''
                entity = ''
                ann_list = []
    print ('total sentences: {0}, total cost: {1}'.format(count, total_cost))
    wf.close()
                    

if __name__ == '__main__':
    
    # 加载模型
    nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05')
    
    parse = argparse.ArgumentParser()
    parse.add_argument('-r', '--rfile path', help='Source file name',
                   dest='rf', required=True)
    parse.add_argument('-w', '--wfile path', help='Result output path',
                        dest='wf', required=True)
    args = parse.parse_args()
    
    if not os.path.exists(args.wf):
        os.mkdir(args.wf)
        
    rf = args.rf.split('/')[-1]
    
    wf = '{0}/{1}_DepencyParsing'.format(args.wf, rf)
    
    print ('正在加载 StanfordCoreNLP 模型')
    DependencyParsing(args.rf, wf)

    # 关闭模型
    nlp.close()