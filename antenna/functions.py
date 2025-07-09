from antenna.utils import *
from antenna.models import *
from antenna.ranger import Ranger
from antenna import *

from torch.optim.optimizer import Optimizer
from abc import ABC, abstractmethod


#%% Import By Device
FloatTensor = torch.FloatTensor if str(config.device) == 'cpu' else torch.cuda.FloatTensor # type: ignore

class SurrogateModel(ABC):
    def __init__(self, model, criterion, optimizer, *, progress_callback = lambda i, n: None):
        """
        Parameters
        ----------
        progress_callback: function 
            A callback function that will be called for every frame to notify
            the saving progress. It must have the signature ::

                def func(current_frame: int, total_frames: int) -> Any

            where *current_frame* is the current frame number and
            *total_frames* is the total number of frames to be saved.
            *total_frames* is set to None, if the total number of frames can
            not be determined. Return values may exist but are ignored.

            Example code to write the progress to stdout::

                progress_callback = lambda i, n: print(f'Saving frame {i}/{n}')
        """
        self.FloatTensor = torch.FloatTensor if str(config.device) == 'cpu' else torch.cuda.FloatTensor # type: ignore
        self.epoch = 1
        self.model: nn.Module = model
        self.criterion: nn.Module = criterion
        self.optimizer: Optimizer = optimizer
        self.progress_callback = progress_callback

    def save(self, rootdir):
        # path = Path(rootdir).joinpath(f"sm_{self.epoch}.pth")
        path = Path(rootdir).joinpath(f"sm.pth")
        torch.save(self.model, path)
        return path

    def load(self, rootdir):
        path = Path(rootdir).joinpath(f"sm.pth")
        self.model = path.load_torch()

    def __call__(self, pattern):
        self.epoch += 1
        return self.model(pattern)
    def __str__(self):
        return f"{self.__class__.__name__}(Model={self.model.__class__.__name__}, Optimizer={self.optimizer.__class__.__name__}, Criterion={self.criterion.__class__.__name__})"
    
    @abstractmethod
    def train(self, pattern):
        pass

class SpecialSM(SurrogateModel):
    def __init__(self):
        model_ge = HFSSNet( # Pattern -> Response
            AntennaPattern.getAllPixel(), config.response_size
        )
        criterion_ge = nn.MSELoss()
        optimizer_ge = Ranger(
            params=model_ge.parameters(), lr=config.lr
        )
        self.scheduler_ge = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_ge, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        super().__init__(model_ge, criterion_ge, optimizer_ge)

    def train(self, pattern):
        self.model.train()
        pattern = tensor(pattern)
        sm_loss = []
        for epoch_ge in range(500):
            self.progress_callback(epoch_ge, 500)
            self.optimizer.zero_grad()

            response = AntennaResponse(self.model(pattern))
            
            match pattern.size(0):
                case 625: #? 25*25
                    s11 = AntennaResponse(response[0])
                    s21 = AntennaResponse(response[1])
                    s22 = AntennaResponse(response[2])

                    loss_s11 = s11.criterion('S11')
                    loss_s21 = s21.criterion('S21')
                    loss_s22 = s22.criterion('S22')
                    
                    loss_ge:Tensor = loss_s11 + loss_s21 + loss_s22
                case 1600: #? 40*40
                    loss_ge:Tensor = AntennaResponse(response).criterion()
                case _:
                    raise ValueError(f'No matching settings found for {pattern.size(0)}')
                
            loss_ge.backward()
            self.optimizer.step()
            self.scheduler_ge.step(loss_ge.item())
            sm_loss.append(loss_ge.item())

        return sm_loss

class OldSM(SurrogateModel):
    def __init__(self):
        model_ge = HFSSNet( # Pattern -> Response
            1600, config.response_size
        )
        criterion_ge = nn.MSELoss()
        optimizer_ge = Ranger(
            params=model_ge.parameters(), lr=config.lr
        )
        self.scheduler_ge = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_ge, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        super().__init__(model_ge, criterion_ge, optimizer_ge)

    def train(self, pattern:Tensor, real_response:Tensor):
        self.model.train()
        pilotLoss_2 = []
        self.loss = float('inf')
        epoch_2 = 0
        
        input = tensor(pattern,  requires_grad=True)
        label = tensor(real_response,  requires_grad=True)
        # for epoch in range(num_epochs):
        while self.loss > 0.00005 and epoch_2 < 2000:
            
            self.optimizer.zero_grad()

            outputs_result:Tensor = self.model(input)

            loss_R:Tensor = self.criterion(
                outputs_result.reshape(-1, *config.response_size),
                label.reshape(-1, *config.response_size)
            )

            loss_R.backward()
            self.optimizer.step()

            pilotLoss_2.append(loss_R.item())
            self.loss = loss_R.item()
            self.progress_callback(epoch_2, 2000)

            epoch_2 = epoch_2 + 1

        return pilotLoss_2


HFSS_model = Path(r"C:\timmy\Program\Antenna\model.pt").load_torch()
