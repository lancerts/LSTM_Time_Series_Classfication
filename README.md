# LSTM/GRU for Time Series Classification





##  2017-06-04
* The code is compatible for python 3.6 and tensorflow 1.1.
* The static RNN is deployed in the post above and we adopt the dynamical RNN in
  tensorflow in the code.
* We further modify the batch process and add the GRU cells.
* For the ChlorineConcentration data set, applying the train-test (10%/90%) split discussed in [this paper,](https://arxiv.org/pdf/1603.06995v4.pdf), it is easy to reach >75% test accuracy.


## Credits
Credits for this project go to [LSTM_tsc](https://github.com/RobRomijnders/LSTM_tsc) for providing a strong example, the [UCR archive](http://www.cs.ucr.edu/~eamonn/time_series_data/) for the dataset.
