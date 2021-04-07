import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score

def update_plot(fig,ax,TL,VL,F1,last_ep_f1,best_f1,topf1_logs,model_name,sample):
    #global comentario
    titulo="Current Training:"+' '+model_name+" Sample="+str(sample)
    [ax_.cla() for ax_ in ax]
    # plot 0 : TL, VL, last_ep
    ax[0].plot(range(len(TL)), TL, lw=2, label='Train Loss')
    ax[0].plot(range(len(VL)), VL, lw=2, label='Valid Loss')
    #ax[0].axvline(last_ep_vl,c='r',marker='|',lw=1,label="Last improve")
    #ax[1].plot(range(len(F1)),np.full_like(F1,max(F1)),'r--',lw=1, label='best F1')
    # plot 1 : F1, topf1_logs
    ax[1].plot(range(len(F1)), F1, lw=2, label='F1 score')
    label_epoch = "Last improve "+str(last_ep_f1)
    ax[1].axvline(last_ep_f1,c='r',marker='|',lw=1,label=label_epoch)
    if len(topf1_logs)==0:
        # plot topf1 actual
        label_f1 = 'top F1 actual '+str('%.3f'%(max(F1)))
        ax[1].plot(range(len(F1)),np.full_like(F1,max(F1)),'r--',lw=1, label=label_f1)

    else:
        # plot topf1 mean
        # revisar ultima epoca subida f1
        label_f1 = 'top F1 '+str('%.3f'%(best_f1))
        ax[1].plot(range(len(F1)),np.full_like(F1,best_f1),'r--',lw=1, label=label_f1)
        ax[1].plot(range(len(F1)),np.full_like(F1,np.mean(topf1_logs)),'g--',lw=1, label='mean F1 hist')
    [ax_.set_xlabel('Epochs') for ax_ in ax]
    [ax_.grid() for ax_ in ax]
    [ax_.legend() for ax_ in ax]
    fig.suptitle(titulo)
    fig.canvas.draw()
    fig.canvas.flush_events()

def training(modelo, criterion, lr,wd, epochs, use_gpu, train_loader, valid_loader,path_full,path_save,fig,ax,topf1_logs,model_name,sample):
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr,weight_decay=wd)
    #ultima_mejora --> VL,
    # ambas por época (no cambian juntas)
    #ultima_mejora = 0
    paciencia = 250
    ultima_subida_f1 = 0
    best_valid,best_f1= np.inf,0.0
    # Historiales por épocas
    TL,VL,F1=list(),list(),list()

    for k in tqdm(epochs):
        if (k-ultima_subida_f1 >= paciencia):
            print("Hace al menos ", paciencia," épocas NO aumenta F1 Score macro. Best F1=",best_f1)
            # guardar plot
            plt.savefig(path_save)
            break

        train_loss, valid_loss,f1_acum = 0.0, 0.0, 0.0
        # Loop de entrenamiento
        for batch in train_loader:
            inputs, labels,period,amp = batch['data'],batch['label'],batch['period'],batch['amplitud']
            if use_gpu:
                inputs, labels,period,amp  = inputs.cuda(), labels.cuda(),period.cuda(),amp.cuda()
            inputs = inputs.transpose(1,2)
            period,amp = period.unsqueeze(-1).float(),amp.unsqueeze(-1).float()
            #display(inputs.shape,labels.shape,period.shape,amp.shape )
            #break
            data,mask = inputs[:,[1,2]],inputs[:,4].unsqueeze(0).transpose(0,1)
            #display(data.shape,mask.shape)
            #display(period,amp)
            #optimizer.zero_grad()
            outputs = modelo.forward(data,mask,period,amp,t=0)
            #display(outputs.shape,labels.shape)
            #break
            loss = criterion(outputs,labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        TL.append(train_loss/train_loader.__len__())

        # VALIDACION
        for batch in valid_loader:
            #actual_f1 = f1_acum/test_loader.__len__()
            inputs, labels,period,amp = batch['data'],batch['label'],batch['period'],batch['amplitud']
            if use_gpu:
                inputs, labels,period,amp  = inputs.cuda(), labels.cuda(),period.cuda(),amp.cuda()
            inputs = inputs.transpose(1,2)
            period,amp = period.unsqueeze(-1).float(),amp.unsqueeze(-1).float()
            #display(inputs.shape,labels.shape,period.shape,amp.shape )
            #break
            data,mask = inputs[:,[1,2]],inputs[:,4].unsqueeze(0).transpose(0,1)

            y_true=labels.cpu().numpy()
            #display(y_true)
            #break
            outputs = modelo.forward(data,mask,period,amp,t=0)
            # F1 score
            y_pred=outputs.cpu().detach().argmax(dim=1).numpy()
            #print(y_pred)
             # cambio 'weighted' por 'macro' (seguro? no será 'micro'?)
            f1_acum += f1_score(y_true, y_pred, average='macro')
            loss = criterion(outputs,labels)
            #print(loss.item())
            valid_loss += loss.item()
            #print(valid_loss)
            # save best model
        # usar F1 como condición para guardar
        f1_last = f1_acum/valid_loader.__len__()
        valid_loss_last = valid_loss/valid_loader.__len__()
        if f1_last > best_f1:
            ultima_subida_f1 = k
            #print("vamos mejorando :) epoch=",k+1)
            #best_valid = valid_loss
            best_f1 = f1_last
            torch.save({'epoca': k,
                        'f1_score': best_f1,
                        'model_state_dict': modelo.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'Valid_loss': valid_loss_last}, path_full)
        VL.append(valid_loss_last)
        F1.append(f1_last)
        update_plot(fig,ax,TL,VL,F1,ultima_subida_f1,best_f1,topf1_logs,model_name,sample)

    # Retornar modelo a la CPU
    if use_gpu:
        modelo = modelo.cpu()
        #print("ok")
    return best_f1
