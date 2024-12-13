import torch.nn as nn
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,fea_in,fea_out):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(fea_in, 8*fea_out),
            nn.BatchNorm1d(8*fea_out),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(8*fea_out, 8*fea_out),
            nn.BatchNorm1d(8*fea_out),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(8*fea_out, 4*fea_out),
            nn.BatchNorm1d(4*fea_out),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(4*fea_out, fea_out),
            nn.BatchNorm1d(fea_out),
            nn.ReLU(True)
        )
    def forward(self, x):
            return self.encoder(x)
    
class PICL(nn.Module):  # 20, 59
    def __init__(self,dim_list,class_num,args):
        super(PICL, self).__init__()

        # list of the linear layer
        self.encoders = nn.ModuleList()
        self.PI_encoders = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        self.class_num = class_num

        for i in range(args.view_num):
            self.encoders.append(Encoder(dim_list[i], args.fea_out))   
            PI_dim = sum(dim_list) - dim_list[i]
            self.PI_encoders.append(Encoder(PI_dim,args.fea_out))   
            classifier = nn.Sequential(
                nn.Linear(args.fea_out, class_num)
            )
            self.classifiers.append(classifier)  
        
            self.register_buffer("prototypes_" + str(i), torch.zeros(class_num, args.fea_out))  
            
   


    def forward(self, X_list,labels,args):

        
        fea, logits, prototypes = [], [], []

        for i, x in enumerate(X_list):
            # query view
            q = self.encoders[i](x.view(x.size()[0], -1))    
            
            # key view
            PI = torch.cat([X_list[j].view(X_list[j].size()[0], -1) for j in range(len(X_list)) if j != i], dim=1)
            k = self.PI_encoders[i](PI)

            features = torch.cat((q, k), dim=0)   
            fea.append(features) 
            logit = self.classifiers[i](q)  
            logits.append(logit)

            # update prototypes
            with torch.no_grad():
                proto = getattr(self, "prototypes_" + str(i))
                for key, label in zip(k,labels):
                    proto[label] = proto[label]*args.proto_m+(1.0 -args.proto_m) * key
                proto = F.normalize(proto, p=2, dim=1)

            prototypes.append(proto)
            
            
        return fea,logits,prototypes
        



        


