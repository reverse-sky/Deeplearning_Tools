# augmentation

사용 

```python
criterion = nn.CrossEntropyLoss() #cut mix를 사용하지 않을 때 쓰는 criterion
CMcriterion = CutMixCriterion() # CUtmix를 사용할 때의 criterion
...
for e in range(0, n_epochs):
    for data, labels in tqdm(train_loader):
        rand_num = np.random.rand() #난수 생성  0~1사이의 probability

        if rand_num > 0.75: data, (target1, target2, lam) = cutmix(data, labels) #1/4확률로 cutmix를 사용하고, 
        elif rand_num > 0.5:  # 1/2의 확률로 mixup을 사용  if문 에 따라서 둘중 하나만 사용이 됨 
            data, target1, target2, lam = mixup_data(data, labels)
            data, target1, target2 = map(Variable, (data, target1, target2))

        data = data.float().to(device) # labels.to(device) # move tensors to GPU if CUDA is available     
        if rand_num > 0.5: labels = (target1.to(device), target2.to(device), lam) 
        else: labels = labels.to(device)

        optimizer.zero_grad() # clear the gradients of all optimized variables

        if rand_num > 0.5: loss = CMcriteion(outputs,labels)
        else: loss = criterion(outputs,labels)



```
