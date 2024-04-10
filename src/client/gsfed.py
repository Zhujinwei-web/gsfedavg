from copy import deepcopy
import torch

from fedavg import FedAvgClient

class gsFedAvgClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)
        self.group_leader_info = {}
        self.global_model = deepcopy(self.model)
        
    @torch.no_grad()
    def aggregate(
        self,
        groupInfo,
        device
    ):
        weights = torch.tensor(groupInfo.weightCache, device) / sum(groupInfo.weightCache)
        delta_list = [list(delta.values()) for delta in groupInfo.deltaCache]
        aggregated_delta = [
            torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
            for diff in zip(*delta_list)
        ]

        for param, diff in zip(groupInfo.dict.values(), aggregated_delta):
            param.data -= diff
        # self.global_model.load_state_dict(groupInfo.dict, strict=False)

