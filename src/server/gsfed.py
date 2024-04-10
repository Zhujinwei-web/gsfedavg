from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
import math
import time
from typing import List

import ctypes

import torch

from src.client.gsfed import gsFedAvgClient

from fedavg import FedAvgServer, get_fedavg_argparser

# 定义 Python 中与 C++ 结构体 GroupInfo 对应的类
class GroupInfo(ctypes.Structure):
    _fields_ = [
        ("gpk", ctypes.c_char_p),  # 假设 gpk 的类型是字符串
        ("1111", ctypes.c_char_p),
        ("2222", ctypes.c_char_p),
        ("gmsk", ctypes.c_char_p), # 假设 gpk 的类型是字符串
        ("1111", ctypes.c_char_p),
        ("2222", ctypes.c_char_p),
        ("gamma", ctypes.c_char_p), # 假设 gpk 的类型是字符串
        ("1111", ctypes.c_char_p),
        ("2222", ctypes.c_char_p),
        ("param", ctypes.c_char_p),
    ]

# 定义 Python 中与 C++ 结构体 GroupInfo 对应的类
class GroupJoin(ctypes.Structure):
    _fields_ = [
        ("param_info", ctypes.c_char_p),  # 假设 gpk 的类型是字符串
        ("gmsk_info", ctypes.c_char_p),
        ("gpk_info", ctypes.c_char_p),
        ("gamma_info", ctypes.c_char_p)
    ]

# 定义 Python 中与 C++ 结构体 GroupSig 对应的类
class GroupSig(ctypes.Structure):
    _fields_ = [
        ("gpk_info", ctypes.c_char_p),
        ("sk_info", ctypes.c_char_p),
        ("param_info", ctypes.c_char_p),
        ("message", ctypes.c_char_p)
    ]

# 定义 Python 中与 C++ 结构体 group_verify 对应的类
class GroupVerify(ctypes.Structure):
    _fields_ = [
        ("sig", ctypes.c_char_p),
        ("message", ctypes.c_char_p),
        ("gpk_info", ctypes.c_char_p),
        ("param_info", ctypes.c_char_p)
    ]

#加载动态链接库
lib = ctypes.CDLL('/usr/local/lib/libgroup_sig.dylib')

# 声明函数签名
# 假设 create_group_default 返回一个 GroupInfo* 的指针
lib.create_group_default.restype = ctypes.POINTER(GroupInfo)
# lib.create_group_default.restype = GroupInfo

# 假设 group_member_join 接受类型和返回类型
# lib.group_member_join.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
lib.group_member_join.argtypes = [ctypes.POINTER(GroupJoin)]
lib.group_member_join.restype = ctypes.c_char_p

# 假设 group_sig 接受类型和返回类型
lib.group_member_join.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
lib.group_member_join.restype = ctypes.c_char_p

# 假设 group_verify 接受类型和返回类型
lib.group_verify.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
lib.group_verify.restype = bool

def get_pfedsim_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--beta", type=float, default=1.0)
    return parser

# 定义 Python 函数来调用 group_member_join
def call_group_member_join(param_info, gmsk_info, gpk_info, gamma_info):
    # 创建 GroupInfo 实例并设置值
    group_info = GroupJoin()
    group_info.param_info = param_info
    group_info.gmsk_info = gmsk_info
    group_info.gpk_info = gpk_info
    group_info.gamma_info = gamma_info
    
    # 调用 C 函数
    # result = lib.group_member_join(group_info.param_info, group_info.gmsk_info, group_info.gpk_info, group_info.gamma_info)
    result = lib.group_member_join(group_info.param_info, group_info.gmsk_info, group_info.gpk_info, group_info.gamma_info)
    return result

# 定义 Python 函数来调用 group_sig
def call_group_sig(gpk_info, sk_info, param_info, message):
    # 创建 GroupInfo 实例并设置值
    group_info = GroupSig()
    group_info.gpk_info = gpk_info
    group_info.sk_info = sk_info
    group_info.param_info = param_info
    group_info.message = message
    
    # 调用 C 函数
    result = lib.group_sig(group_info)
    return result

# 定义 Python 函数来调用 group_verify
def call_group_verify(sig, message, gpk_info, param_info):
    # 创建 group_verify 实例并设置值
    group_info = GroupVerify()
    group_info.sig = sig
    group_info.message = message
    group_info.gpk_info = gpk_info
    group_info.param_info = param_info
    
    # 调用 C 函数
    result = lib.group_verify(group_info)
    return result

class gsFedServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "gsFed",
        args: Namespace = None,
        unique_model=False,
        default_trainer=True,
    ):
        if args is None:
            args = get_pfedsim_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = gsFedAvgClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
        self.test_flag = False
        perGroupMembers = len(self.train_clients) / 5
        self.group_member_sig = {}
        self.adminGroup = None
        self.group_info = []
        for index, client in enumerate(self.train_clients):
            if (index % perGroupMembers == 0):
                ##管理员群签名
                if (index/perGroupMembers == 0):
                    self.adminGroup = lib.create_group_default()
                # 加群
                clientInfo = lib.group_member_join(self.adminGroup.contents.param, self.adminGroup.contents.gmsk, self.adminGroup.contents.gpk, self.adminGroup.contents.gamma)
                # 成员群签名
                lib.group_sig(self.adminGroup.contents.gpk, clientInfo, self.adminGroup.contents.param, 'success'.encode('utf-8'))
                ##分群
                self.group_info.append({
                    'currentId': client,
                    'groupParam': lib.create_group_default(),
                    'dict': self.global_params_dict,
                    'deltaCache': [],
                    'weightCache': []
                })
            currentGroupInfo = self.group_info[math.floor(index/perGroupMembers)]['groupParam'].contents
            # 加群
            clientInfo = lib.group_member_join(currentGroupInfo.param, currentGroupInfo.gmsk, currentGroupInfo.gpk, currentGroupInfo.gamma)
            # 成员群签名
            self.group_member_sig[client] = lib.group_sig(currentGroupInfo.gpk, clientInfo, currentGroupInfo.param, 'success'.encode('utf-8'))
    
    def train(self):
        """The Generic FL training process"""
        avg_round_time = 0
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log("-" * 26, f"TRAINING EPOCH: {E + 1}", "-" * 26)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]
            begin = time.time()
            self.train_one_round()
            if E == self.args.global_epoch - 1:
                self.aggregate()
            end = time.time()
            self.log_info()
            avg_round_time = (avg_round_time * (self.current_epoch) + (end - begin)) / (
                self.current_epoch + 1
            )
        self.logger.log(
            f"{self.algo}'s average time taken by each global epoch: {int(avg_round_time // 60)} m {(avg_round_time % 60):.2f} s."
        )
            
    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            (
                delta,
                weight,
                self.client_stats[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_local_params,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )
            current_group_index = next((index for index, i in enumerate(self.group_info) if lib.group_verify(str(self.group_member_sig[client_id]).encode('utf-8'), 'success'.encode('utf-8'), i['groupParam'].contents.gpk, i['groupParam'].contents.param)), None)
            if current_group_index is not None:
                self.group_info[current_group_index].deltaCache.append(delta)
                self.group_info[current_group_index].weightCache.append(weight)
            else:
                # 处理current_group_index为None的情况
                pass
        for client_id in self.selected_clients:
            if lib.group_verify(str(self.group_member_sig[client_id]).encode('utf-8'), 'success'.encode('utf-8'), self.adminGroup.contents.gpk, self.adminGroup.contents.param):
                index = next((i for i, info in enumerate(self.group_info) if info['currentId'] == client_id), None)
                self.trainer.aggregate(self.group_info[index], self.device)
                
    @torch.no_grad()
    def aggregate(
        self
    ):
        weight_cache = []
        delta_cache = []
        for i in range(len(self.group_info)):
            weight_cache.extend(self.group_info[i]['weightCache'])
            delta_cache.extend(self.group_info[i]['deltaCache'])
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)
        delta_list = [list(delta.values()) for delta in delta_cache]
        aggregated_delta = [
            torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
            for diff in zip(*delta_list)
        ]

        for param, diff in zip(self.global_params_dict.values(), aggregated_delta):
            param.data -= diff
        self.model.load_state_dict(self.global_params_dict, strict=False)    

if __name__ == "__main__":
    server = gsFedServer()
    server.run()
