## TODO: The original dataset consists of 80k 2M-sample traces and the dataset used in most side-channel attack papers seems to be condensed down to a 5k-trace dataset with 4k used for training, 500 for validation, and 500 for testing. It is not clear how the author decided to extract these traces or the subset of relevant samples for each trace. It seems like the approach was to take traces corresponding to a single key value for both the training and testing datasets, which would seem extremely problematic and likely render results meaningless.

import datasets

