#Important notes :
1. batch size is number of data the model has to process before updating its parameters. If the batch size is very small, the model takes very less time to update its parametes. In other words, the model looks into the very fewer data of the entire datasets and hence noise will be more. Alternatively, if the batch size is large means, the model is looking into the large data or it has a bigger picture of data and hence noise will be reduced. But the update will very slow (means low speed). Noise is inversely proportional to the batch size.
2. The best way to acess the data from the dataloaders is to use iter() and next()
3.   ######################################################
      data_iter = iter(dataloader)  # Creates an iterator object. The iterator data_iter knows how to fetch batches sequentially from the dataloader.
     ####################################################
      first_batch = next(data_iter)  # Get the first batch
   The next() function asks the iterator for the next batch. Since batch_size=2, the first batch contains the first 2 samples from the dataset.
