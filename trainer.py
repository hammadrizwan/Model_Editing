from imports import *
from models import SiameseClassificationNetwork,SiameseNetwork,SiameseNetworkTriplet
from loss_functions import WeightedContrastiveLoss,CosineSimilarityLoss_SentenceTransformers,ContrastiveLoss,TripletLoss
import data_processing as dp

def train_model_combined(model,optimizer,data_loader,criterion_similarity,
                         criterion_classification=None,mode="simlarity",
                         path_to_folder="./",file_name="best_model_weights.pth",epochs=50,early_stop_patience=5,device="cpu"):
  # try:
  num_epochs = epochs
  lowest_error = float('inf')
  best_model_weights = None
  counter_early_stop=0
  print("num_epochs",num_epochs)
  for epoch in range(num_epochs):
      total_loss = 0.0
      total_batches = len(data_loader)
      for batch in data_loader:
          embs1, embs2, labels, _ , _, _,_ = batch
          optimizer.zero_grad()
          if(mode=="classificaiton"):
            output1, output2, output3 = model(embs1,embs2)
            loss_classification = criterion_classification(output3, labels)
            labels_cosine = torch.where(labels == 0, torch.tensor(0.5).to(device), torch.tensor(0.9).to(device))
            loss_semantic_sim = criterion_similarity(output1,output2, labels_cosine)
            alpha = 0.0
            beta = 1.0
            combined_loss = (alpha * loss_classification) + (beta * loss_semantic_sim)
            combined_loss.backward()
          else:
            output1, output2 = model(embs1,embs2)
            combined_loss = criterion_similarity(output1, output2, labels)#semantic sim loss
            combined_loss.backward()

          optimizer.step()

          total_loss += combined_loss.item()
      # Calculate average loss after the epoch
      epoch_loss = total_loss / total_batches
      

      # Check for early stopping
      if epoch_loss < lowest_error:
          lowest_error = epoch_loss
          best_model_weights = model.state_dict()
          counter_early_stop = 0  # Reset the counter when there is an improvement
      else:
          counter_early_stop += 1
      print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f} Early Stop: {counter_early_stop}')
      # Check for early stopping
      if counter_early_stop >= early_stop_patience:
          print(f'Early stopping triggered at epoch {epoch + 1} as the loss did not improve.')
          break
  torch.save(best_model_weights, path_to_folder+file_name)
  state_dict = torch.load(path_to_folder+file_name)
  model.load_state_dict(state_dict)
  print('Training finished.')
  return model,path_to_folder+file_name



def train_control(dataset_paired_train,loss_function,learning_rate=0.0001,weight_decay=0.01,path_to_folder="./",epochs=50,cls_weights=True,device="cpu",early_stop_patience=5):
  """
    loss_function: loss function to be used cosine, cosine_crossentropy, contrastive
    learning_rate: learning rate for Adam optimizer
  """

  if(cls_weights):# compute class weights
    labels=[]
    train_data_loader=dp.get_data_loader(dataset_paired_train,batch_size=1,shuffle=False,device=device)
    for row in train_data_loader:
      labels.append(row[2].item())
    class_weights = compute_class_weight(class_weight='balanced',classes=np.array(np.unique(labels)), y=np.array(labels))
    print(class_weights)
  else:
    class_weights=[1.0,1.0]

  
  if(loss_function=="cosine" or  loss_function=="cosine_crossentropy"):
    train_data_loader=dp.get_data_loader(dataset_paired_train,batch_size=512,shuffle=True,device=device)
    if(loss_function=="cosine_crossentropy"):
      mode="classificaiton"
      class_weights=torch.tensor(class_weights,dtype=torch.float)
      criterion_classification = nn.CrossEntropyLoss(weight=class_weights).to(device)
      model=SiameseClassificationNetwork(512).to(device)
    else:
      criterion_classification=None
      mode="similarity"
      model=SiameseNetwork(512).to(device)
    criterion_similarity = CosineSimilarityLoss_SentenceTransformers().to(device)
    comparison="sim"
  elif(loss_function=="contrastive"):
    print("inside contrastive")
    train_data_loader=dp.get_data_loader(dataset_paired_train,batch_size=512,shuffle=True,device=device)
    model=SiameseNetwork(512).to(device)
    mode="similarity"
    comparison="dist"
    #12.0=MARGIN
    criterion_similarity=WeightedContrastiveLoss(margin=50.0, positive_weight=class_weights[1], negative_weight=class_weights[0]).to(device)
    criterion_classification=None
  else:
    train_data_loader=dp.get_data_loader_triplet(dataset_paired_train,batch_size=512,shuffle=True,device=device)
    model=SiameseNetworkTriplet(512).to(device)
    mode="similarity"
    comparison="dist"
    #12.0=MARGIN
    criterion_similarity = nn.TripletMarginLoss(margin=10.0, p=2, eps=1e-7).to(device)
    criterion_classification=None
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)# for cosine use 0.00001 else 0.0001
    model,file_path=train_model_triplet(model,optimizer,train_data_loader,criterion_similarity,path_to_folder,"best_model_weights.pth",early_stop_patience=early_stop_patience,device=device,epochs=epochs)
    return model,mode,comparison,file_path
  
  optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)# for cosine use 0.00001 else 0.0001
  model,file_path=train_model_combined(model,optimizer,train_data_loader,criterion_similarity,criterion_classification,mode,path_to_folder,"best_model_weights.pth",early_stop_patience=early_stop_patience,device=device,epochs=epochs)
  return model,mode,comparison,file_path


def train_model_triplet(model,optimizer,data_loader,criterion,path_to_folder="./",file_name="best_model_weights.pth",early_stop_patience=5,device="cpu",epochs=50):
    num_epochs = epochs
    lowest_error = float('inf')
    best_model_weights = None
    early_stop_patience=early_stop_patience
    for epoch in range(num_epochs):
        total_batches = len(data_loader)
        total_loss=0
        for batch in data_loader:
            embs1, embs2, embs3, _ , _, _,_,_ = batch
            optimizer.zero_grad()

            output1, output2, output3 = model(embs1,embs2,embs3)
            combined_loss = criterion(output2,output3, output1)#semantic sim loss
            combined_loss.backward()

            optimizer.step()

            total_loss += combined_loss.item()
        # Calculate average loss after the epoch
        epoch_loss = total_loss / total_batches

        # Check for early stopping
        if epoch_loss < lowest_error:
            lowest_error = epoch_loss
            best_model_weights = model.state_dict()
            counter_early_stop = 0  # Reset the counter when there is an improvement
        else:
            counter_early_stop += 1

        print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f} Early Stop: {counter_early_stop}')
      # Check for early stopping
        if counter_early_stop >= early_stop_patience:
            print(f'Early stopping triggered at epoch {epoch + 1} as the loss did not improve.')
            break
    torch.save(best_model_weights, path_to_folder+file_name)
    state_dict = torch.load(path_to_folder+file_name)
    model.load_state_dict(state_dict)
    print('Training finished.')
    return model,path_to_folder+file_name