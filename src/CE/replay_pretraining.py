"""
The replay pretraining mechanism works by
forcing the model to see its training sequence
twice. This forces it to remember and learn from
the tokens that pass by, rather than just predict.
"""

from typing import List, Tuple, Optional
import random

def apply_replay_formatting(
                    token_sequence: List[int],
                    bos_token_id: int,
                    sep_token_id: int,
                    eos_token_id: int,
                    truncate_length: Optional[int],
                    )->List[int]:
    """
    Sets up replay pretraining for a token
    sequence, based on the given sequence, and
    inserting the ids
    :param token_sequence: The token sequence to
    :param bos_token_id: beginning of sequence id
    :param sep_token_id: sep sequence id
    :param eos_token_id: end of sequence id
    :return: The sequence with replay pretraining in place
    """
    if truncate_length is not None and len(token_sequence) > truncate_length:
        token_sequence = token_sequence[:truncate_length]
    token_sequence = ([bos_token_id] +
                      token_sequence +
                      [sep_token_id] +
                      token_sequence +
                      [eos_token_id])
    return token_sequence

class ReplayStreamFormatter:
    """
    A small stateful class that automatically
    does scheduling from the provided stream
    inputs. Invoke it with the tokenized steam
    entries and the schedule to have it
    automatically truncate and apply replay
    formatting. Drawing from the dataset advances
    the schedule, and the formatter stops truncating
    when its data runs out.
    """
    def __init__(self,
                 eos_token_id: int,
                 bos_token_id: int,
                 sep_token_id: int,
                 schedule: List[Tuple[int, int]],
                 rand_min: float = 0.8
                 ):
        """
        :param bos_token_id: beginning of sequence id
        :param sep_token_id: sep sequence id
        :param eos_token_id: end of sequence id
        :param schedule: The schedule. A list of tuples of ints
                First number is how many examples it lasts.
                Second is what length to truncate
        :param rand_min: The minimum value, between 0..1, to let the
               random modification to truncate length decrease to. Used
               to prevent the model from learning the truncation length.
        """
        self.bos_token_id = bos_token_id
        self.sep_token_id = sep_token_id
        self.eos_token_id = eos_token_id

        self.schedule = schedule + [(None, None)]
        self.rand_min = rand_min

        remaining_schedule, truncate_length = self.schedule.pop(0)
        self.truncate_length = truncate_length
        self.remaining_schedule = remaining_schedule

    def __call__(self, token_sequence: List[int]) -> List[int]:
        """
        Processes and returns the formatted replay string
        :param token_sequence: The token sequence
        :return: The replay formatted token sequence
        """
        if self.remaining_schedule is not None:
            truncate_length = int(self.truncate_length*random.uniform(self.rand_min, 1.0))
            self.remaining_schedule -= 1
            if self.remaining_schedule == 0:
                schedule_length, new_truncate_length = self.schedule.pop(0)
                self.remaining_schedule = schedule_length
                self.truncate_length = new_truncate_length
        else:
            truncate_length = None
        return apply_replay_formatting(token_sequence,
                                       self.bos_token_id,
                                       self.sep_token_id,
                                       self.eos_token_id,
                                       truncate_length)


