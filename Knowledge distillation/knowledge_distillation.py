import torch
import torch.nn as nn
import torch.nn.functional as F

def knowledge_distillation_loss(teacher_logits,student_logits,labels,criterion = F.cross_entropy):
    alpha = 0.1
    T = 10

    student_loss = criterion(input=student_logits,target = labels)  # Calculate the loss about the Student model
    distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_logits/T, dim=1), F.softmax(teacher_logits/T, dim=1)) * (T * T)
    total_loss =  alpha*student_loss + (1-alpha)*distillation_loss
    return total_loss

def knowledge_distillation_train(teacher_model,student_model,loader,device,criterion,optimizer,display = True,amp= True):
    train_loss = []
    train_acc  = []
    scaler =  GradScaler()
    teacher_model.to(device)
    student_model.to(device)

    if display:
        bar = tqdm(loader)
    else:
        bar = loader
    teacher_model.eval()
    student_model.train()
    for i, (data,target) in enumerate(bar):
        optimizer.zero_grad()
        data,target = data.to(device),target.to(device)
        with autocast():     #torch.float32 -> torch.float16
            teacher_logits = teacher_model(data)
            student_logits = student_model(data)
            total_loss = knowledge_distillation_loss(teacher_logits,student_logits,target,criterion)
        
        scaler.scale(total_loss).backward() #amp를 사용하려면 scaler를 사용해야함 
        scaler.step(optimizer)
        scaler.update()
        train_loss.append(total_loss.detach().cpu().numpy())
        
        ps = F.softmax(student_logits, dim=1)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == target.reshape(top_class.shape)
        train_acc.append(torch.mean(equals.type(torch.float)).detach().cpu())
        
        if display:bar.set_description('Train Loss: {:.5f} \tAcc: {:.5f}'.format(np.mean(train_loss),(np.mean(train_acc))))
    
        
        
        
    train_loss = np.mean(train_loss)  # mean loss of one epoch
    train_acc  = np.mean(train_acc)   # mean acc  of one epohc
    return train_loss ,train_acc    



def knowledge_distillation_val(teacher_model,student_model,loader,device,criterion,optimizer,display = True,amp= True):
    val_loss = []
    val_acc  = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    scaler =  GradScaler()
    teacher_model.to(device)
    student_model.to(device)

    if display:
        bar = tqdm(loader)
    else:
        bar = loader
    teacher_model.eval()
    student_model.eval()
    with torch.no_grad():
        for i, (data,target) in enumerate(bar):
            optimizer.zero_grad()
            data,target = data.to(device),target.to(device)
            with autocast():    
                teacher_logits = teacher_model(data)
                student_logits = student_model(data)
                total_loss = knowledge_distillation_loss(teacher_logits,student_logits,target,criterion)
            
            probs = student_logits.softmax(dim =1)                         # 다중분류 -> 각 클래스일 확률을 전체 1로 두고 계산하기

            LOGITS.append(student_logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())        
            
            ps = F.softmax(student_logits,dim=1)
            top_p,top_class = ps.topk(1,dim=1)
            equals = top_class==target.reshape(top_class.shape)

            
            val_loss.append(total_loss.detach().cpu().numpy())
            val_acc.append(torch.mean(equals.type(torch.float)).detach().cpu())
            
            if display:bar.set_description('Valid Loss: {:.5f} \tAcc: {:.5f}'.format(np.mean(val_loss),(np.mean(val_acc))))    
            
    #list로 되어있는 것을 cat하여 한번에 계산 
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    val_acc = (PROBS.argmax(1) == TARGETS).mean()
    # accuracy : 정확도
    val_loss = np.mean(val_loss)
    # val_acc = np.mean(val_acc)
    return val_loss, val_acc ,PROBS,TARGETS  


