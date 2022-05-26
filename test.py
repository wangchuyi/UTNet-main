import numpy 
import torch



if __name__ == '__main__':
    # crossentropyloss=torch.nn.CrossEntropyLoss()
    # x_input = torch.tensor([[0,0,0,1]],dtype=torch.float32)
    # y_target=torch.tensor([3])
    # y_target=y_target.long()
    # print(y_target.dtype)
    # crossentropyloss_output=crossentropyloss(x_input,y_target.long())
    # print("???",crossentropyloss_output)
    # print(torch.nn.functional.one_hot(y_target))
    x_input=torch.zeros((1,4,2,2))
    x_input[:,-1,:,:]=1
    y_target=3*torch.ones((1,2,2)).long()
    print(x_input,y_target)
    y_target =  torch.nn.functional.one_hot(y_target)
    y_target = y_target.permute(0, 3, 1,2) 


    mse = torch.nn.MSELoss(reduction='mean')
    mse_loss = mse(x_input,y_target)
    crossentropyloss=torch.nn.CrossEntropyLoss()
    crossentropyloss_output=crossentropyloss(x_input,y_target)
    
    print('crossentropyloss_output:\n',crossentropyloss_output)