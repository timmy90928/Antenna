from typing import (
    Tuple, List, Dict, Deque, # Can use the built-in.
    TypeVar, cast, Callable, Any, Optional, overload, Union, Sequence
)
from typing_extensions import Self
import traceback
from torch import (
    __version__,
    nn,
    tensor as _tensor,
    Tensor,
    cuda,
    manual_seed as _manual_seed,
    load as _torch_load,
    save as _torch_save,
    device as _torch_device,
    # get_default_device,
    set_default_device,
    stack,
    concat
)
from numpy import (
    ndarray,
    random
)
from  pickle import (
    dump as _pickle_dump,
    load as _pickle_load
)
from json import (
    load as _json_load, 
    dump as _json_dump
)
from pandas import DataFrame
from tqdm import trange
from collections import defaultdict
from warnings import filterwarnings

from pathlib import Path as _Path
from os.path import getctime

import numpy as np
from copy import deepcopy
from datetime import datetime
from shutil import rmtree as _rmtree

#* Figure
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes._axes import Axes  # type: ignore

#* Email
from smtplib import SMTP
from email.mime.text import MIMEText


FIG_CONFIG = {
    "format": 'png',
    "bbox_inches": "tight",
    "pad_inches": 0.1,
    "dpi": 300,
    "transparent": True,
    "facecolor": "none", # white
    "edgecolor": "none",
}

try: 
    from torch.utils.tensorboard import SummaryWriter # type:ignore pip install tensorboard
    def getTensorBoardWriter(log_dir:str = './runs') -> SummaryWriter:
        """

        ## Usage
        ```bash
        tensorboard --logdir=runs
        ```

        ## Example
        ```
        tbwriter = getTensorBoardWriter()
        for n_iter in range(100):
            tbwriter.add_scalar('Loss/train', np.random.random(), n_iter)
            tbwriter.add_scalar('Loss/test', np.random.random(), n_iter)
            tbwriter.add_scalar('Accuracy/train', np.random.random(), n_iter)
            tbwriter.add_scalar('Accuracy/test', np.random.random(), n_iter)
        ```
        """
        return SummaryWriter(log_dir)
except ModuleNotFoundError:
    pass

def errorCallback(errorCallback:Optional[Callable[[str],Any]]=None, *errorCallbackArgs, **errorCallbackKwargs):
    """
    Error callback function.

    ## Usage
    ```python
    @errorCallback()
    def func():
        raise Exception("Error")
    ```
    """
    def decorator(func:Callable):
        def wrap(*args, **kwargs):
            try:
                return func(*args, **kwargs)   # print(func.__name__)
            except Exception as e:
                if errorCallback:
                    errorCallback(str(e), *errorCallbackArgs, **errorCallbackKwargs)
                else:
                    print(e)
        return wrap
    return decorator

class Path(type(_Path()), _Path): # type: ignore
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)
    
    def __init__(self, path: str, create:bool=False):
        """
        Path model.
       
        ## Usage
        ```python
        path = Path("./path/to/file.ext")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        path.unlink()
        path.del_from_glob('*.pth')
        path.manage_file_count('*.pth', keep_latest=3)
        path.load_torch()
        path.not_exist_create(create_file=True)
        ```
        """
        # self.path = path
        
        if create: self.not_exist_create()

    def rmtree(self) -> bool:
        if self.is_dir():
            _rmtree(self)
            return True
        else:
            return False
    
    def not_exist_create(self, create_file:bool = False):
        """
        Create the path if it does not exist.

        :param create_file: Whether to create the file.
        :return: Whether the path does not exist.
        """
        if self.suffix:
            self.parent.mkdir(parents=True, exist_ok=True)
            if create_file: self.touch(exist_ok=True)
        else:  # No file extension, treated as a directory.
            self.mkdir(parents=True, exist_ok=True)
        return self

    def del_from_glob(self, pattern:str):
        """
        Delete all files matching the pattern.

        :param pattern: Patterns matching files, E.g., '*.pth'
        """
        if not self.suffix:
            paths = list(self.glob(pattern))
            for path in paths:
                path.unlink()
        else:
            self.unlink()

    def manage_file_count(self, pattern:str, keep_latest:int = 3):
        """
        Manage the number of archives and only keep the latest specified number.

        :param pattern: Patterns matching archives, E.g., '*.pth'
        :param keep_latest: Latest quantity to keep.
        """

        # Confirm that the target directory exists.
        if not self.exists():
            raise FileNotFoundError(f"The destination directory ({self.absolute()}) does not exist.")
        
        # Get all files matching the pattern.
        files_sorted = sorted(self.glob(pattern), key=getctime)

        # If the file exceeds the limit, delete the oldest file.
        if len(files_sorted) > keep_latest:
            for old_backup in files_sorted[:len(files_sorted)-keep_latest]:
                if not old_backup.rmtree():
                    old_backup.unlink()
            return True
        else:
            return False
    
    def load_torch(self, device = None):
        from antenna.models import config
        if __version__ >= "2.6.0":
            return _torch_load(self, weights_only=False, map_location=device or config.device)
        else:
            return _torch_load(self, map_location=device or config.device)

_PATHLIKE = Union[str, Path]

class LossFunction:
    """
    Loss function class.
    
    Attributes:
        label: Label of the loss function.
        loss_function: Loss function.

        is_early_stopping: Whether to use early stopping.
        early_stop: Whether to stop the training.
        best_model: The best model.
        
    

    """
    _loaded:Dict[str, "LossFunction"] = {}
    def __new__(cls, loss_function:Callable, label:str = "", *args, **kwargs):
        label = label or loss_function.__name__
        if cls._loaded.get(label):
           client = cls._loaded.get(label)
           assert isinstance(client, cls)
        else:
           client = super().__new__(cls)  
           cls._loaded[label] = client

        return client
            
    @overload
    def __init__(self, loss_function:Callable[[Tensor, Tensor], Tensor], label:str = ""):...
    @overload
    def __init__(self, loss_function:Callable[[], Tensor], label:str = ""):...
        
    def __init__(self, loss_function:Callable, label:str = ""):
        """
        Loss function class.

        :param loss_function: Loss function.
        :param label: Label of the loss function.
        
        ## Usage::
        ```python
        loss_function = LossFunction(nn.MSELoss(), "MSE")
        loss_function.enableEarlyStopping(nn.Module, patience=10, delta=0.001)
        loss_function(output, target)

        loss_function.plot()
        loss_function.save_as_numpy("loss.npy")
        loss_function.load_from_numpy("loss.npy")
        ```

        """
        label = label or loss_function.__name__
        self.label = label
        self.loss_function = loss_function

        self._loss_record = getattr(self, "_loss_record", [])

        ###* EarlyStopping ###
        self.is_early_stopping:bool =  getattr(self, "is_early_stopping", False)
        """Whether to use early stopping."""
        self.early_stop:bool =  getattr(self, "early_stop", False)
        """Whether to stop the training."""
        self.best_model = getattr(self, "best_model", None)
        """The best model."""
        
    
    def __call__(self, output:Optional[Tensor] = None, target:Optional[Tensor] = None, *, record:bool = True) -> Tensor:
        """
        Call the loss function.

        Args:
            output: Output of the model.
            target: Target of the model.
            record: Whether to record the loss.

        Returns:
            Loss value.
        """
        if output is None and target is None:
            _loss:Tensor = self.loss_function()
        else:
            _loss:Tensor = self.loss_function(output, target)
        
        if self.is_early_stopping:
            self._earlyStopping(_loss)

        if record: self._loss_record.append(_loss.item())
        return _loss
    
    def enableEarlyStopping(self, model:nn.Module, patience:int = 10, delta:float = 0.001, *, verbose:bool = False, trace_func=print):
        """
        Enable early stopping for the model.

        :param patience: Number of epochs with no improvement after which training will be stopped.
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.
        :param verbose: If True, prints a message for each validation loss improvement.
        :param trace_func: Function used to print messages.
        
        """
        self.model = getattr(self, "model", model)
        self._patience = getattr(self, "_patience", patience)
        self._delta = getattr(self, "_delta", delta)

        assert self.model, "Model not defined!"
        assert self._patience > 0, "Patience should be greater than 0"
        assert self._delta > 0, "Delta should be greater than 0"
        
        self._counter:int = getattr(self, "_counter", 0)
        self._best_score = getattr(self, "_best_score", None)
        self.val_loss_min =  getattr(self, "val_loss_min", np.Inf)
        self._verbose = getattr(self, "_verbose", verbose)
        self._trace_func = getattr(self, "_trace_func", trace_func)

        self.is_early_stopping = True

    def _earlyStopping(self, val_loss:Tensor):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        """
        score = -val_loss.item()

        if self._best_score is None:
            self._best_score = score
            self._save_early_stopping_checkpoint(val_loss)
        elif score < self._best_score + self._delta:
            self._counter += 1
            if self._verbose: self._trace_func(f'EarlyStopping counter: {self._counter} out of {self._patience}')
            if self._counter >= self._patience:
                self.early_stop = True
        else:
            self._best_score = score
            self._save_early_stopping_checkpoint(val_loss)
            self._counter = 0

    def _save_early_stopping_checkpoint(self, val_loss):
        if self._verbose:
            self._trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        self.best_model = deepcopy(self.model.state_dict())
        self.val_loss_min = val_loss

    def save_as_numpy(self, path:_PATHLIKE):
        np.save(str(path), self._loss_record)
        return path
    
    def load_from_numpy(self, path:_PATHLIKE):
        self._loss_record = np.load(str(path))
        return self
    
    def plot(self, axes:Optional[Axes] = None, show:bool = False):
        ax:Axes = plt.axes(axes) # type: ignore
        ax.set_title(f'Loss Function ({self.loss_function.__name__})')
        ax.plot(self._loss_record)
        # ax.legend()
        if show: plt.show()
        return ax
    
def plot(x,file_name:Optional[str] = None) -> None:
        """
        Plot the weight matrix on a 3D graph
        """
        # This part is for plotting the graph
        plt.clf()
        # plt.figure(figsize=(20, 10))
        plt.title(f'')
        plt.plot(x)
        plt.legend()

        plt.show()

        if file_name: plt.savefig(file_name, **FIG_CONFIG)

class Config(dict):
    def __init__(self):
        self.element_num = 40
        self.epochs = 10
        self.incTH_deg , self.refTH_deg = -40 , 20 #先不要改
        self.count = 0      # An integer count value
        self.Main_lr = 1e-3 # Learning Rate For Main Training Loop

        self._checkpoint_save_path = Path("./checkpoint")
        
    @property
    def device(self):
        return _torch_device(type='cpu')
    
    @device.setter
    def device(self, device):
        device = device or _torch_device("cuda:0" if cuda.is_available() else "cpu")
        set_default_device(device)
        if device != "cpu":
            cuda.set_device(self.device)
    
    @property
    def checkpoint_save_path(self):
        """
        ```
        config.checkpoint_save_path.not_exist_create()
        ```
        """
        return self._checkpoint_save_path.absolute()
        
    @checkpoint_save_path.setter
    def checkpoint_save_path(self, path):
        self._checkpoint_save_path = Path(path)

    def setRandomSeeds(self, seed = 0):
        _manual_seed(seed)
        cuda.manual_seed(seed)
        random.seed(seed)
    
    def setWarning(self, warning_type:str = "ignore"):
        return filterwarnings(warning_type) # type: ignore
    

config = Config()

def tensor(data: Any,dtype= None, device=None, requires_grad: bool = False):
    return _tensor(data, dtype=dtype, device=device or config.device, requires_grad=requires_grad)

class Figure:
    def __init__(self, name:str, nrowcol:tuple = (1, 1), save:bool = False, show:bool = False, rootdir:Optional[str] = None, **kwargs):
        """
        ## Example
        ```
        with Figure("test_3_2", nrowcol=(2,2), save=True) as fig:
    
            ax1 = fig.index(1)
            ax1.set_title("test")
            ax1.plot([1, 2, 3, 4])

            fig.addAll()
            fig[2].set_title("test")
            fig[2].plot([1, 2, 3, 4])
        ```
        
        ## Set
        ```
        class:
            ...
            def plot(self, axes:Axes|None = None):
                ax:Axes = plt.axes(axes) # type: ignore
                ax.set_title("test")
                ax.plot([1, 2, 3, 4])
        ```

        ## 動畫
        with Figure("asd", (1, 2)) as fig:
            fig.fig.set_size_inches(9, 6)
            fig.addAll()
            loss_list = []
            def update(frame):
                fig[0].clear()
                fig[1].clear()
                fig[0].set_title("Loss")
                fig[0].set_xlim(0, epochs)
                fig[0].set_ylim(0, 2.5)
                loss_list.append(r["loss"][frame])
                fig[0].plot(loss_list)
                fig[1].set_title("Generated Pattern")
                fig[1].imshow(r["output"][frame], cmap='gray')
                return fig
            fig.saveGIF(update, epochs, dpi=150)
        """
        fig = plt.figure(name, **kwargs)
        # fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        self.fig = fig
        self.save = save
        self.show = show
        self.name = name
        self.nrowcol = nrowcol    
        self.rootdir = Path(rootdir or "./")
        

    def index(self, index:int = 1):
        ax = self.fig.add_subplot(self.nrowcol[0], self.nrowcol[1], index)
        return ax
    
    def addAll(self):
        for i in range(self.__len__()) :
            self.index(i+1)

    def saveGIF(self, update:Callable, epochs:int = 10, dpi = 150):
        writer = PillowWriter(fps=30, metadata={"artist": "WeiWen Wu"})
        tqdm_iter = trange(epochs, desc="Plotting")
        ani = FuncAnimation(self.fig, update, frames=epochs)
        ani.save(f"{self.rootdir.joinpath(self.name)}.gif", writer=writer, dpi=dpi, progress_callback=lambda i, n: tqdm_iter.update())
    
    def __getitem__(self, index:int) -> Axes:
        """
        Use first
        ```
        fig.addAll()
        ```
        """
        return self.fig.get_axes()[index]

    def __len__(self) -> int:
        return self.nrowcol[0] * self.nrowcol[1]

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback):
        FIG_CONFIG = {
            "format": 'png',
            "bbox_inches": "tight",
            "pad_inches": 0.1,
            "dpi": 300,
            "transparent": True,
            "facecolor": "white", # white or none
            "edgecolor": "white", # white or none
        }
        
        if not exc_type:
            if self.show: plt.show()
            if self.save: 
                # self.fig.set_size_inches(18, 12)
                plt.savefig(self.rootdir.joinpath(f"{self.name}.png"), **FIG_CONFIG)  
 
        plt.close()

class Record:
    def __init__(self, name:str = "record", rootdir:Optional[str] = None, load:bool = False):
        self._data = defaultdict(list)
        self._history = defaultdict(list)
        self.name = name
        self.path = Path(rootdir or "./").joinpath(
            f"{name}.record"
        )

        if load: self.load()
    
    def __call__(self, key, default = None, *, append = False):
        """Get the last value of key."""
        return self.end(key, default, append = append)

    def __setitem__(self, key, value):
        self._data[key].append(value)
                                
    def __getitem__(self, key):
        """Get the complete array of keys."""
        if self.__contains__(key):
            return self._data[key]
        else:
            _keys = ', '.join(self._data.keys())
            raise KeyError(f"{key} does not exist. (Current key: {_keys})")
        
    def  __contains__(self, item:str):
        return item in self._data

    def end(self, key, default = None, *, append = False):
        if self.__contains__(key):
            return self.__getitem__(key)[-1]
        else:
            if append:
                self.__setitem__(key, default)
                return self.end(key)
            else:
                return default
        
    def add(self, key, num, default = None):
        """
        add('a', 1):
        a += 1
        """
        self.__setitem__(
            key, self.end(key, default) + num
        )
        

    
    def save(self, description:Optional[str] = None):
        self._history["time"].append(str(datetime.now()).split(".")[0])
        self._history["description"].append(description or "No description")
        self._history["len"].append(len(self))

        with open(str(self.path), "wb") as f:
            _pickle_dump(
                {
                    "_data": self._data,
                    "_history": self._history
                }, 
                file = f
            )
    
    def load(self):
        if not self.path.exists():
            self.save()
        with open(str(self.path), "rb") as f:
            _loaded = _pickle_load(f)
            self._data = _loaded["_data"]
            self._history = _loaded["_history"]
        
        return self._data
    
    def average(self, key):
        return sum(self._data[key]) / len(self._data)
    
    def same(self, key, value, *, start:int = 0, end:int = -1):
        if isinstance(value, ndarray):
            return  any(
                np.array_equal(value, x) 
                for x in self._data[key][start:end]
            )

        return value in self._data[key][start:end]
    
    def early_stop(self, key: str, patience: int = 10) -> bool:
        """
        根據指定 key 的歷史資料，決定是否應該 early stop。
        若最近 `patience` 次都沒有改善，回傳 True。
        """
        values = self._data[key]
        if len(values) < patience + 1:
            return False  # 數據不足，不應該停止

        best = min(values[:-patience])
        recent = values[-patience:]

        if all(v <= best for v in recent):
            return True
        return False

    def reset(self):
        self._data = defaultdict(list)
    
    def custom(self, key:str, fn:Callable):
        return fn(self._data[key])
    
    @property
    def dataframe(self):
        try:
            return DataFrame(self._data)
        except ValueError as e:
            raise ValueError(f"{e}\n{repr(self)}")

    @property
    def history(self):
        return DataFrame(self._history)

    def __str__(self):
        return str(self.dataframe)
    
    def __repr__(self):
        _str = ''
        for key, value in self._data.items():
            _str += f"{key}[{len(value)}] "

        return f"Record({self.name}: {_str})"
    
    def __len__(self):
        return len(self.dataframe)
    
class json:
    """
    ### Example
    ```
    from utils.utils import json
    _json = json('static/config.json')
    print(_json('base/UPLOAD_FOLDER'))
    _json_data = _json.load()
    print(_json_data['success'])
    _json_data['success'] = False
    _json.dump(_json_data)
    ```
    """
    def __init__(self, path:str, create:bool = True) -> None:
        self.path = Path(path)
        
        if not self.path.exists():
            if create:
                self.path.touch()
                self.dump({})
            else:
                raise FileNotFoundError(f"JSON file '{path}' does not exist.")

    @overload
    def __call__(self, key:str) -> Any: 
        """
        Get the value of the specified key in the JSON file.

        ### Example
        >>> _json('base/UPLOAD_FOLDER')
        """
    ...
    @overload
    def __call__(self, key:str, value:Any) -> dict: 
        """
        Set the value of the specified key in the JSON file.

        ### Example
        >>> _json('base/UPLOAD_FOLDER', 'new/path')
        """
    ...
    def __call__(self, key:str, value = None):
        keys = key.split('/')
        if value is not None: 
            if value == "null": value = None
            if value in ["True", "true"]: value = True
            if value in ["False", "false"]: value = False
            result = self._set(keys, value)
            self.dump(result)
            return result
        else:
            return self._get(keys)
    def __getitem__(self, key):
        return self.__call__(key, value = None)
    def __setitem__(self, key, value):
        return self.__call__(key, value)
    
    def get(self, key:str, default = None):
        keys = key.split('/')
        try:
            return self._get(keys)
        except KeyError:
            result = self._set(keys, default)
            self.dump(result)
            return default

    def _set(self, keys:list, value:Any) -> dict:
        temp =  self.load().copy()
        _ = "temp"
        for i, k in enumerate(keys):
            if k == '': continue
            _ += f"['{k}']"

            if i == len(keys) - 1:
                exec(f"{_} = value")
            else:
                if k not in temp:
                    exec(f"{_} = {{}}")

        return temp
        
    def _get(self, keys:list) -> Any:

        self.data = self.load()
        result = self.data.copy()
        for k in keys:
            if k == '': continue
            result = result[k]
        return result
    def load(self) -> dict:
        with open(self.path, 'r', encoding='utf-8') as f:
            return _json_load(f)

    def dump(self, data:dict) -> bool:
        with open(self.path, 'w', encoding='utf-8') as f:
            _json_dump(data, f, ensure_ascii=False, indent=4)
        return True

    def delete(self, key:str) -> bool:
        keys = key.split('/')
        data = self.load()
        
        # Traversing through the keys
        temp = data
        for k in keys[:-1]:  # Get to the parent of the key to delete
            if k in temp:
                temp = temp[k]
            else:
                return False  # If the key doesn't exist, return False
        
        # Deleting the key
        if keys[-1] in temp:
            del temp[keys[-1]]
            self.dump(data)  # Save the updated data back to the file
            return True
        else:
            return False

class Email(SMTP):
   
    def __init__(
            self, 
            to_addr:Union[str, Sequence[str]],
            from_addr_pwd:tuple = ("ailab@ee.ccu.edu.tw", "bung ovhd rrcu nayg")
        ) -> None:
        """
        Example
        -------
        ```
        with Email("weiwen@alum.ccu.edu.tw") as email:
            msg = email.getText("This is a test email sent from Python.")

            msg['Subject'] = 'test測試' # 郵件標題
            msg['From'] = 'AI Lab'  # 暱稱 或是 email
            msg['To'] = 'weiwen@alum.ccu.edu.tw'    # 收件人 email 或 暱稱
            msg['Cc'] = 'weiwen@alum.ccu.edu.tw, XXX@gmail.com'   # 副本收件人 email 
            msg['Bcc'] = 'weiwen@alum.ccu.edu.tw, XXX@gmail.com'  # 密件副本收件人 email

            status = email.sendMessage(msg.as_string())
            
            if status == {}:
                print("Email sent successfully!")
            else:
                print('Email send failed!')
        ```

        Reference
        ---------
        https://steam.oxxostudio.tw/category/python/example/gmail.html
        """
        super().__init__("smtp.gmail.com", 587)
        self.starttls()
        self.login(from_addr_pwd[0], from_addr_pwd[1])
        
        self.to_addr = to_addr
        self.from_addr = from_addr_pwd[0]
    
    def getText(self, message):
        return MIMEText(message)
                        
    def sendMessage(self, message):
        return self.sendmail(self.from_addr, self.to_addr, message)

    def __enter__(self) -> Self:
        return self
    
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb) -> None:
        self.quit()

if __name__ == "__main__":
    # print(Path("./checkpoint").manage_file_count("*.pth", keep_latest=1))
    # print(Path("./checkpoint/GEN_model_0.pth").load_torch())
    config.device = 'cpu'
    r = Record('Temp', load=True, rootdir=r"D:\patch_result\1750340068")

    print(r)
    print(r.history)
    # r.save()
    # loss = LossFunction(CustomLoss())
    # for i in range(10):
    #     loss(tensor([i]), tensor([i + i]))
    # loss.plot(show=True)
