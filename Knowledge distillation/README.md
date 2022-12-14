#  Knoledge Distillation 
## Knoeldge Distillation의 이론은 간단하다.
Teacher model(사전에 잘 학습이 된 모델)을 바탕으로 Student model(연산량이 낮은 모델)이 학습을 진행하는 것이다.
> 이로 얻을 수 있는 장점은 Limited environment에서의 연산량을 줄일 수 있으며, 파라메터의 사이즈도 훨씬 적기에 edge computing이 가능하다는 장점을 가진다. 
#### 자세한 것은 [knowledge distillation이 왜 학습에 좋은가?](https://www.microsoft.com/en-us/research/blog/three-mysteries-in-deep-learning-ensemble-knowledge-distillation-and-self-distillation/)참조
[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

Loss: ![image](https://user-images.githubusercontent.com/45085563/207545304-c8b30b4f-a63e-45e4-ad1f-57fff6a984bd.png)  

Loss function 구현  [코드 구현 참조](https://re-code-cord.tistory.com/entry/Knowledge-Distillation-1)
```python
def knowledge_distillation_loss(teacher_logits,student_logits,labels,criterion = F.cross_entropy):
    alpha = 0.1
    T = 10

    student_loss = criterion(input=student_logits,target = labels)  # Calculate the loss about the Student model
    distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_logits/T, dim=1), F.softmax(teacher_logits/T, dim=1)) * (T * T)
    total_loss =  alpha*student_loss + (1-alpha)*distillation_loss
    return total_loss
```

-----------
# In train code

```python
  #train
import knowledge_distillation
criterion = nn.CrossEntropyLoss()
...
  teacher_model.eval() # Student model의 train과정에서 teacher모델은 이미 학습이 완료된 상태, 학습을 진행하면 안된다. 
  student_mdoel.train()
  for data,labels in train_loader:
    optimizer.zero_grad()
    teacher_outputs = teacher_model(data)
    student_outputs = student_model(data)
    total_loss = knowledge_distillation_loss(teacher_outputs,student_outputs,labels,criterion)
    total_loss.backward()  #backpropagation about the total loss
    ## Student models are updated according to rules 
    ##that reduce losses between teacher and student output.
    optimzer.step()
    
```
