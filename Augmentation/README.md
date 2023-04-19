# augmentation

사용 

```python
"""
just add the cut out or mixup code according to probability
"""

criterion = nn.CrossEntropyLoss() # criterion when the probability lower than 0.5
...
for e in range(0, n_epochs):
    for data, labels in tqdm(train_loader):
        prob = np.random.rand() # Create the Random Numver between 0~1, then it will be probability

        # Using the Cutmix if probability is bigger than 0.75 
        if prob > 0.75: data, label= cutmix(data, labels) 
        # Or using the mixup when over the 1/2 probability
        elif prob > 0.5: data, label= mixup(data, labels)
                
        data,labels = data.to(device),labels.to(device)
        
        optimizer.zero_grad() # clear the gradients of all optimized variables
        
        logits = model(data)
        loss = criterion(logits,labels)
        
        # if prob > 0.5: loss = CMcriteion(outputs,labels)
        # else: loss = criterion(outputs,labels)

```
