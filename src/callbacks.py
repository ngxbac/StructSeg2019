from catalyst.dl.core import Callback, CallbackOrder
from catalyst.dl.utils import criterion
from catalyst.dl.core.state import RunnerState
from catalyst.utils import get_activation_fn


class MultiDiceCallback(Callback):
    """
    Dice metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "dice",
        activation: str = "Sigmoid",
        num_classes : int = 7,
    ):
        """
        :param input_key: input key to use for dice calculation;
            specifies our `y_true`.
        :param output_key: output key to use for dice calculation;
            specifies our `y_pred`.
        """
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.activation = activation
        self.num_classes = num_classes

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]

        activation_fnc = get_activation_fn(self.activation)
        outputs = activation_fnc(outputs)

        dice = 0
        for cls in range(self.num_classes):
            targets_cls = (targets == cls).float()
            outputs_cls = outputs[:, cls]
            score = criterion.dice(outputs_cls, targets_cls, eps=1e-7, activation='none', threshold=None)
            dice += score / self.num_classes
            state.metrics.add_batch_value(name=self.prefix + str(cls), value=score)
        state.metrics.add_batch_value(name=self.prefix, value=dice)
