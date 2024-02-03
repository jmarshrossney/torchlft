from torchlft.nflow.train import ReverseKLTrainer
from torchlft.nflow.logging import Logger as BaseLogger


class Trainer(ReverseKLTrainer):
    def _logging_step(self, model, step):
        fields, actions = model(1000)
        #print(fields.inputs.var(), fields.outputs.var())
        return super().logging_step(model, step)

class Logger(BaseLogger):
    pass

