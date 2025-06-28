from antenna.utils import *
from script.get_local_ip import getLocalIP
from antenna.patch import com_error
# import numpy as np

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from loguru import logger #? pip3 install loguru

import sys

# __all__ = ["AntennaPattern", "AntennaResponse", "GradientEstimator", "SPGEN"]


def mult(_ob):
    _result = 1
    for i in _ob:
        _result *= i
    return _result

class AntennaResponse:
    """
    Antenna Response Design.

    """
    _target_response = {}
    _loss_fn_hook = {}
    def __init__(self, response:Tensor):
        """
        Antenna Response Design.

        Args:
            response (Tensor): Response of the antenna.
    
        Return:
            AntennaResponse
        
        Raises:
            TypeError: If the response is not a tensor.
        
        """
        if not isinstance(response, Tensor):
            raise TypeError("Expected Tensor, but got {}".format(type(response)))
        response = response.to(config.device)

        if len(response.shape) == 1:
            self.response = response
            self.vertical = self._reshape2vertical()
        else:
            self.response = response.reshape(-1)
            self.vertical = response

    def __invert__(self):
        """Detach the response"""
        return self.response.detach().cpu()
    
    def _reshape2vertical(self):
        assert len(self.response.shape) == 1
        _v = self.response.reshape(1, self.response.shape[0])
        _v.requires_grad_(True)
        return _v
    
    @property
    def loss(self) -> LossFunction:
        if not hasattr(self, '_loss'):
            self._loss = LossFunction(self.lossResponse, label="Response Loss")
        return self._loss
    
    def lossResponse(self):
        target = self.getTargetResponse().vertical  # [361]
        prediction = self.vertical  # [361]

        # 基本條件
        mask_20 = target == -20
        mask_b_20 = prediction[mask_20] > -20

        mask_0 = target == 0
        mask_s_0 = prediction[mask_0] < -3

        # 為了確保有梯度，設定條件不滿足時也會計入一個 dummy loss
        if mask_b_20.sum() > 0:
            loss_20 = F.smooth_l1_loss(
                prediction[mask_20][mask_b_20],
                target[mask_20][mask_b_20]
            )
        else:
            # 使用全體 prediction 的一小部分作 dummy loss，保證梯度
            loss_20 = 0.01 * F.mse_loss(prediction, target)

        if mask_s_0.sum() > 0:
            loss_0 = F.smooth_l1_loss(
                prediction[mask_0][mask_s_0],
                target[mask_0][mask_s_0]
            )
        else:
            loss_0 = 0.01 * F.mse_loss(prediction, target)

        loss = loss_20 + loss_0

        return loss

    
    
    def plot(self, label, axes:Optional[Axes] = None, show:bool = False):
        ax:Axes = plt.axes(axes) # type: ignore
        ax.set_title(f'Antenna Response')
        ax.plot(self.getTargetResponse(label).response.cpu().detach(), color='red', label='Target')
        ax.plot(self.response.cpu().detach(), color='blue', label='Simulation')
        ax.legend()
        if show: plt.show()
        return ax
    
    # @classmethod
    # def setTargetResponse(cls, _min:int, _width:Tuple[int,int,int,int,int], label:Optional[str] = None) -> "AntennaResponse":
    #     """
    #     Target Response Design.

    #     :param _min: The lowest point of response
    #     :param _width: 

    #     :return: AntennaResponse
        
    #     """
    #     if len(_width) != 5:
    #         raise ValueError(f"Expected 5 width, but got {len(_width)}")
    #     setattr(cls, '_target_response_min', _min)
    #     setattr(cls, '_target_response_width', _width)

    #     return cls.getTargetResponse()
    
    @classmethod
    def registerTargetResponse(cls, side:float, center:float, width:Tuple[int,int,int,int,int], label:str = "response") -> Tensor:
        """
        Target Response Design.

        :param side: 
        :param center: 

        :return: AntennaResponse
        
        """
        if len(width) != 5:
            raise ValueError(f"Expected 5 width, but got {len(width)}")
        mask_up = np.concatenate([
            np.ones(width[0]) * side,
            np.linspace(side, center, width[1]),
            np.ones(width[2]) * center,
            np.linspace(center, side, width[3]),
            np.ones(width[4]) * side
        ])
        # expected_response = np.array(mask_up)#.reshape(-1, sum(_width))
        expected_response = tensor(np.array(mask_up), dtype=torch.float32, device=config.device)

        if label:
            cls._target_response[label] = expected_response
        

        return expected_response

    @classmethod
    def getTargetResponse(cls, label:str = "response"):
        """
        Target Response Design.

        Use `setTargetResponse()` before use, otherwise use the default value

        """
        if label not in cls._target_response.keys():
            raise RuntimeError(f"The {label} of TargetResponse is not registered. Please use `registerTargetResponse()` first.")
        return cls._target_response[label]
    
    @classmethod
    def registerLossHook(cls, loss_hook:Callable[[Tensor,Tensor], Tensor], label:str = "response"):
        """
        :param loss_hook: ```def criterion(response, target_response):...``` 
        """
        cls._loss_fn_hook[label] = loss_hook

    def criterion(self, label:str = "response", **param) -> Tensor:
        
        if label not in self._loss_fn_hook.keys():
            raise RuntimeError(f"The {label} of LossHook is not registered. Please use `registerLossHook()` first.")
        
        return self._loss_fn_hook[label](
            self.response, self.getTargetResponse(label), **param
        )
    
class AntennaPattern:
    _history_datas:List[List[torch.Tensor]] = []
    _best_loss = float('inf')
    
    def __init__(self, pattern:torch.Tensor, coordinate:Optional[List[Tuple[int,int, int, int]]] = None):
        """
        Example:
        ```
        AntennaPattern.setCoordinate([(0, 25, 0, 25)])
        ```
            
        """
        self.input_tensor = torch.clamp(pattern.to(config.device), min=0.0, max=1.0)
        coordinate = coordinate or getattr(self, '_antenna_pattern_coordinate')

        self.coordinate:Union[List[Tuple[int,int, int, int]], List] = coordinate or []
        self.num_patterns:int = len(self.coordinate)
        self.num_pixel:int = 0
        self.patterns:List[Tuple[torch.Tensor, int, int, int, int]] = [] # [(pattern, x1, x2, y1, y2), ...] >>> pattern is 2D
        self.series:torch.Tensor = tensor([]) # concatenate: 1-dimensional array 1*n
        """1*n array"""
        
        _shape = self.input_tensor.shape
        if self.dim() == 1:
            self.series = self.input_tensor.reshape(1, _shape[0]) if len(_shape) == 1 else self.input_tensor[0]
        else:
            self.series = self.input_tensor.reshape(1, mult(_shape))
        self._add2patterns()
        
    @classmethod
    def register_simulator(cls, simulator:Callable[[Tensor],Dict[str, Tensor]]):
        cls._simulator = simulator

    @classmethod
    def getAllPixel(cls):
        """
        TODO: 目前是取回所有的像素點，但實際上是取得大圖的像素點
        """
        if hasattr(cls, '_antenna_pattern_coordinate'):
            coordinates:List[Tuple[int, int, int, int]] = getattr(cls, '_antenna_pattern_coordinate')
        else:
            raise ValueError("No antenna pattern coordinate found, `AntennaPattern.setCoordinate()`")
        _result = 0
        for x1, x2, y1, y2 in coordinates:
            _size = (x2 - x1, y2 - y1)
            _result += mult(_size)
        return _result
    
    @classmethod
    def getRandomPattern(cls, w=40, h=40):
        patterns = torch.randn(
            w,h, 
            dtype=torch.float32,
            device=config.device
        )
        binaries = (patterns > 0.5).float()
        return cls(binaries, [(0, w, 0, h)])

    def __getitem__(self, key) -> "AntennaPattern":
        if key >= self.__len__():
            raise IndexError(f"Expected size {self.__len__()} but got size {key}")
        pattern, x1, x2, y1, y2 = self.patterns[key]
        return AntennaPattern(pattern, [(x1, x2, y1, y2)])
    
    def __add__(self, other):
        if isinstance(other, AntennaPattern):
            _pattern = torch.cat([self.series, other.series], dim=1)
            _coordinate = self.coordinate + other.coordinate
            
            return AntennaPattern(_pattern, _coordinate)
        else:
            raise TypeError("Unsupported operand type for +: 'AntennaPattern' and '{}'".format(type(other)))
    
    def __len__(self):
        return len(self.patterns)
    
    def __invert__(self):
        """Detach the response"""
        return self.series.detach().cpu()
    
    def dim(self) -> int:
        if len(self.input_tensor.shape) == 1 or self.input_tensor.shape[0] == 1:
            return 1
        else:
            if self.num_patterns == 1:
                return self.input_tensor.dim()
            else:
                raise ValueError("num_patterns > 1 but input_tensor is 1D")      
            
    def copy(self, detach:bool = True):
        if detach:
            return AntennaPattern(self.input_tensor.detach().clone(), self.coordinate.copy())
        else:
            return AntennaPattern(self.input_tensor.clone(), self.coordinate.copy())
    
    def _add2patterns(self):
        """
        將 input_tensor 拆分成 self.patterns
        - 如果 num_patterns == 1, 則直接將 input_tensor 作為一個 pattern
        - 如果 num_patterns > 1, 則將 input_tensor 左右拆分成 num_patterns 個 pattern
        """
        _temp = self.series.clone()
        for x1, x2, y1, y2 in self.coordinate:
            _size = (x2 - x1, y2 - y1)
            self.num_pixel += mult(_size)
            pattern = _temp[0:mult(_size)].reshape(_size)
            _temp = _temp[mult(_size):]
            self.patterns.append((pattern, x1, x2, y1, y2))

    @classmethod
    def setCoordinate(cls, _coordinate:List[Tuple[int, int, int, int]]):
        """
        Coordinate Design.

        """
        if not isinstance(_coordinate, list):
            raise TypeError(f"Expected list, but got {type(_coordinate)}")
        if not all(isinstance(i, tuple) for i in _coordinate):
            raise TypeError(f"Expected list of tuple, but got {type(_coordinate)}")
        if not all(len(i) == 4 for i in _coordinate):
            raise ValueError(f"Expected tuple of length 4, but got {len(_coordinate)}")
        setattr(cls, '_antenna_pattern_coordinate', _coordinate)

    def merge(self) -> torch.Tensor:
        """
        將所有 pattern 合併成一個大的底層 pattern
        - 後加入的 pattern 會覆蓋前面的 pattern
        - 返回合併後的二維 tensor
        """
        if not self.patterns:
            raise ValueError("No patterns to merge")

        max_x = max(x2 for _, _, x2, _, _ in self.patterns)
        max_y = max(y2 for _, _, _, _, y2 in self.patterns)
        base_pattern = torch.zeros((max_x, max_y), dtype=self.input_tensor.dtype)

        for pattern, x1, x2, y1, y2 in self.patterns:
            base_pattern[x1:x2, y1:y2] = pattern  # 後面的 pattern 覆蓋前面的

        return base_pattern.to(config.device)
    

    def simulate(self, no_grad:bool = False, **param) -> dict[str, AntennaResponse]:
        pattern = self.merge()
        result_response = {}
       
        if hasattr(self, "_simulator"):
            if no_grad:
                with torch.no_grad():
                    result:Dict = self._simulator(pattern, **param)
            else:
                result:Dict = self._simulator(pattern, **param)
        else:
            raise RuntimeError("Please use `register_simulator()` to register the simulator.")
        
        for key, value in result.items():
            result_response[key] = AntennaResponse(value)

        # TODO 
        # if not any([pattern.equal(p) for p, _ in self._history_datas]):
        AntennaPattern._history_datas.append(
            [pattern, result_response]
        )

        return result_response

    
    def plot(self, axes:Optional[Axes] = None, show:bool = False):
        ax:Axes = plt.axes(axes) # type: ignore
        ax.set_title("Antenna Pattern")
        ax.imshow(self.merge().cpu().detach(), cmap='viridis')
        if show: plt.show()
        return ax
    
    def plot_individual(self, axes:Optional[Axes] = None, show:bool = False):
        if not self.patterns:
            raise ValueError("No patterns to merge")

        max_x = max(x2 for _, _, x2, _, _ in self.patterns)
        max_y = max(y2 for _, _, _, _, y2 in self.patterns)
        base_pattern = torch.zeros((max_x, max_y), dtype=self.input_tensor.dtype)
        _result = []
        for pattern, x1, x2, y1, y2 in self.patterns:
            _pattern = base_pattern.clone()
            _pattern[x1:x2, y1:y2] = pattern  
            _result.append(_pattern)

        ax:Axes = plt.axes(axes) # type: ignore
        ax.set_title("Antenna Pattern Individual")
        ax.imshow(torch.cat(_result, dim=1).cpu().detach(), cmap='viridis')
        if show: plt.show()
        return ax

def reshape(_tensor:torch.Tensor):
    _shape = _tensor.shape
    if len(_shape) == 1:
        return _tensor.reshape(1, _shape[0])
    else:
        return _tensor.reshape(_shape[0], 1)


def global_exception_handler(exc_type:type[BaseException] | None, exc_value: BaseException | None, exc_traceback):
    """
    這段是用來擷取Global For Logger.
    ```
    import sys
    sys.excepthook = global_exception_handler
    ```
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    elif issubclass(exc_type, com_error):
        logger.exception(
            f"[{exc_type.__name__}] {exc_value}", 
            exc_inof = (exc_type, exc_value, exc_traceback)
        )
        text = f"這是一個由HFSS ({exc_type.__name__}) 發出的錯誤, 其Error Code為{exc_value.hresult}, 錯誤訊息為{exc_value.strerror}"
    else:
        text = f"這是一個 {exc_type.__name__} 的錯誤, 錯誤訊息為{exc_value}"

    with Email("weiwen@alum.ccu.edu.tw") as email:
        tb_str = '\n'.join(traceback.format_tb(exc_traceback))
        msg = email.getText(f"{text}, 詳細錯誤訊息如下所示\n{tb_str}")

        msg['Subject'] = f'Antanna Error ({getLocalIP()})' 
        msg['From'] = 'AI Lab' 
        msg['To'] = 'weiwen@alum.ccu.edu.tw' 

        status = email.sendMessage(msg.as_string())
            
        if status == {}:
            print("Email sent successfully!")
        else:
            print('Email send failed!')

if __name__ == "__main__":
    config.device = 'cpu'
    # ap = tensor(np.random.rand(40*40), dtype=torch.float32)
    # binary_ap = (ap >= 0.5).float()
    # response = AntennaResponse.getTargetResponse()
    # pattern = AntennaPattern(binary_ap, [(0, 40, 0,40)])
    # pattern.plot()
    # response.plot()
    response = AntennaResponse(torch.randn(361))
    response.plot()





