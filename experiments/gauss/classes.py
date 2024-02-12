from torchlft.nflow.train import ReverseKLTrainer

class Trainer(ReverseKLTrainer):
    def logging_step(self, model, step):
        #error = model.mask * model.weight - model.cholesky
        #rms_error = error.pow(2).sum().sqrt().float()
        #abs_max_error = error.abs().max().float()
        #print(rms_error, abs_max_error)
        return super().logging_step(model, step)
