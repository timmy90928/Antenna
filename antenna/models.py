from antenna.utils import *
from antenna import *

import torch
from torch.autograd.function import (
    Function,
    BackwardCFunction,
)
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer

import numpy as np
from math import sqrt





class GEN(nn.Module):

    def __init__(self ,pattern_pixel):
        super(GEN,self).__init__()
        self.fc_patch = nn.Sequential(
            nn.Linear(pattern_pixel, 2048),
            nn.PReLU(),
            nn.Linear(2048, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512, 361)
        )

        self.to(config.device)
    
    def forward(self, input):
        x = self.fc_patch(input)
        x = x.reshape(-1, 361)
        return AntennaResponse(x)
    
class BiScaleNorm(nn.Module):
    def __init__(self):
        super(BiScaleNorm, self).__init__()

    def forward(self, input_vector):
        # 大於 0 的值的正規化
        max_val = torch.max(input_vector)
        positive_normalized = torch.where(input_vector > 0, input_vector / max_val, torch.tensor(0.0, device=input_vector.device))

        # 小於 0 的值的正規化
        min_val = torch.min(input_vector)
        negative_normalized = torch.where(input_vector < 0, input_vector / torch.abs(min_val), torch.tensor(0.0, device=input_vector.device))

        # 合併正規化結果
        normalized_vector = positive_normalized + negative_normalized
        return normalized_vector

class sign_f(Function):
    """
    sign function
    """
    @staticmethod
    def forward(ctx:BackwardCFunction, inputs:Tensor):
        output = inputs.new(inputs.size())
        output[inputs >= 0.] = 1
        output[inputs < 0.] = -1
        ctx.save_for_backward(inputs)
        return output

    @staticmethod
    def backward(ctx:BackwardCFunction, grad_output:Tensor):
        input_, = ctx.saved_tensors
        grad_output[input_>1.] = 0
        grad_output[input_<-1.] = 0
        return grad_output

class HFSSNet(nn.Module):

    def __init__(self, num_classes):
        super(HFSSNet, self).__init__()
        self.fc_patch = nn.Sequential(
            nn.Linear(625, 2048),
            nn.PReLU(),
            nn.Linear(2048, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 51)
        )

    def forward(self, input):
        x = self.fc_patch(input)
        x = x.reshape(-1, 3, 17)
        return x
    

# %%

class SPGEN(nn.Module):
    def __init__(self ,pattern_table:Tuple, size=40):
        super(SPGEN,self).__init__()

        self.pattern_table = pattern_table
        self.pattern_table_tensor = self._to_tensor()
        self.num_patterns = len(pattern_table) # [Channels] How many small patterns.
        self.grid_size = size // self.patern_size # [big_h, big_w] Composition of small patterns (|===|---|---|---|)
        self.logits = nn.Parameter( 
            #? [batch, big_h, big_w, Channels]
            torch.randn(1, self.grid_size, self.grid_size, self.num_patterns),
            requires_grad=True
        )



    def __str__(self):
        return f"<SPGEN total[{self.patern_size*self.grid_size}(small[{self.patern_size}]xbig[{self.grid_size}])]>"
    
    def _to_tensor(self):
        """
        >>> torch.Size([Channels, small_h * small_w])
        """
        _reselt = []
        for pattern in self.pattern_table:
            self.patern_size: int = len(pattern)
            _reselt.append(np.array(pattern, dtype=np.int16).reshape(-1))

        return torch.tensor(
            np.stack(_reselt), 
            dtype = torch.float32
        )

    def forward(self):

        # 使用 Gumbel-Softmax 模擬 hard selection，但仍可反向傳播
        # hard=True 代表輸出 one-hot（離散選擇）
        # logits: [1, grid_h, grid_w, num_patterns]
        gumbel_probs = F.gumbel_softmax(self.logits, dim=-1, hard=True)  # [1, H, W, P]

        # pattern_table_tensor: [P, small_h * small_w]
        # 對每個位置做 weighted sum（實際是 one-hot 選中的）
        selected_patterns = torch.matmul(gumbel_probs, self.pattern_table_tensor)  # [1, H, W, small_h*small_w]

        # 將每個小圖案 reshape 回 2D 並拼接成一張大圖
        soft_output = selected_patterns.view(
            1, self.grid_size, self.grid_size,  # batch, H, W,
            self.patern_size, self.patern_size  # 小圖案大小
        ).permute(
            0, 1, 3, 2, 4  # (batch, grid_h, small_h, grid_w, small_w)
        ).reshape(
            1,
            self.grid_size * self.patern_size,
            self.grid_size * self.patern_size
        )

        self.output_image = soft_output
        return self.output_image
    
    def show(self, nrowcol:tuple):
        with Figure("SPGEN Small Pattern", nrowcol, show=True) as fig:
            for n in range(len(self)):
                ax:Axes = fig.index(n+1)
                ax.set_title(f"Small Pattern {n+1}")
                ax.imshow(self[n], cmap='viridis')

    def __getitem__(self, idx):
        return self.pattern_table[idx]
    
    def __len__(self):
        return len(self.pattern_table)
    
class PhaseGenModel(nn.Module):
    """
    Generator Model
    """
    def __init__(self , pattern_pixel):
        super(PhaseGenModel,self).__init__()
        self.fc_patch = nn.Sequential(
            nn.Linear(361, 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, pattern_pixel),
            BiScaleNorm(),
        )

        self.r = sign_f.apply
        self.to(config.device)

    def forward(self, input):
        x = self.fc_patch(input)
        x = self.r(x) / 2 + 0.5 # type: ignore
        return AntennaPattern(x)
        # return x

class GradientEstimator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.PReLU(),
            nn.Linear(2048, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512, output_dim)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Flatten()
            
        )
        self.to(config.device)

    def forward(self, A:Tensor):
        A = A.unsqueeze(0) #? [batch, W, H]
        # print(A.shape)
        output = self.conv(A)
        output = self.net(output)
        return AntennaResponse(output)
    
###* Train HFSS model function ###
def train_HFSS_model(patch_pattern:List[Tensor], HFSS_result:List[Tensor], epoch, HFSS_model_name:Path):
    NUM_CLASSES = 2*17 #沒用

    # Train
    init_lr_2 = 0.001

    if epoch == 0:
        HFSS_model = GEN(AntennaPattern.getAllPixel())
        HFSS_model_name = Path("./")
    else:
        HFSS_model = HFSS_model_name.load_torch()
    
    _patch_pattern = torch.stack(patch_pattern, dim=0)
    _HFSS_result = torch.stack(HFSS_result, dim=0)
    
    inputs_2 = Variable(_patch_pattern.type(FloatTensor))   # type: ignore
    labels_2 = Variable(_HFSS_result.type(FloatTensor))     # type: ignore

    HFSS_model.train()

    criterion = nn.MSELoss()
    
    # Optimizer setting
    optimizer_HFSS = torch.optim.Adam(params=HFSS_model.parameters(), lr=init_lr_2)
    
    flag_correct = True
    
    pilotLoss_2 = []
    
    
    
    epoch_2 = 0
    
    # for _epoch in tqdm(range(1200), leave=False):
    
    while (flag_correct):
        
        HFSS_model.train()

        training_loss_2 = 0.0
        

        optimizer_HFSS.zero_grad()
        
        outputs_result:AntennaResponse = HFSS_model(inputs_2)
        
        loss_R:Tensor = criterion(outputs_result.vertical, labels_2.reshape(-1,361)) #?
        
        loss_R.backward()
        optimizer_HFSS.step()
        training_loss_2 += float(loss_R.item() * inputs_2.size(0))
        
        pilotLoss_2.append(training_loss_2)
        
        HFSS_model.eval()
        
        
        if training_loss_2 < 0.0001 or epoch_2 == 800:
            HFSS_model_name = config.checkpoint_save_path.joinpath(f"GEN_model_{epoch}.pth")
            
            torch.save(HFSS_model, str(HFSS_model_name))
            config.checkpoint_save_path.manage_file_count(f"GEN_model_*.pth", 2)
            
            
            flag_correct = False
            break
        
        epoch_2 = epoch_2 + 1
    return HFSS_model_name
