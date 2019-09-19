from typing import Mapping, Any
from catalyst.dl.runner import WandbRunner
from catalyst.dl.core import RunnerState
from catalyst.contrib.optimizers import Lookahead


class ModelRunner(WandbRunner):
    def predict_batch(self, batch: Mapping[str, Any]):
        output = self.model(batch["images"])
        return {
            "logits": output
        }

    # def _prepare_state(self, stage: str):
    #     migrating_params = {}
    #     if self.state is not None:
    #         migrating_params.update({
    #             "step": self.state.step,
    #             "epoch": self.state.epoch + 1
    #         })
    #
    #     self.model, criterion, optimizer, scheduler, self.device = \
    #         self._get_experiment_components(stage)
    #
    #     optimizer = Lookahead(optimizer, k=5, alpha=0.5)
    #
    #     self.state = RunnerState(
    #         stage=stage,
    #         model=self.model,
    #         device=self.device,
    #         criterion=criterion,
    #         optimizer=optimizer,
    #         scheduler=scheduler,
    #         **self.experiment.get_state_params(stage),
    #         **migrating_params
    #     )
