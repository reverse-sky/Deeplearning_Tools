# augmentation

사용 

```python
criterion = nn.CrossEntropyLoss() # criterion when the probability lower than 0.5
CMcriterion = CutMixCriterion() # criterion when the probabiliry over the 0.75
...
for e in range(0, n_epochs):
    for data, labels in tqdm(train_loader):
        rand_num = np.random.rand() # Create the Random Numver between 0~1, then it will be probability

        if rand_num > 0.75: data, (target1, target2, lam) = cutmix(data, labels) # Using the Cutmix if probability is bigger than 0.75 
        elif rand_num > 0.5:  # Or using the mixup when over the 1/2 probability 
            data, target1, target2, lam = mixup_data(data, labels)
            data, target1, target2 = map(Variable, (data, target1, target2))

        data = data.float().to(device) # labels.to(device) # move tensors to GPU if CUDA is available     
        if rand_num > 0.5: labels = (target1.to(device), target2.to(device), lam) 
        else: labels = labels.to(device)

        optimizer.zero_grad() # clear the gradients of all optimized variables

        if rand_num > 0.5: loss = CMcriteion(outputs,labels)
        else: loss = criterion(outputs,labels)



```
