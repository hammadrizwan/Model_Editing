from imports import Dataset,torch,np,random,DataLoader,util
# DATASET CLASSES PYTORCH 
import helper_functions as hp

class CustomDatasetTriplet(Dataset):
    def __init__(self, dataset):
        self.input_sentences1=[val[3] for val in dataset]
        self.input_sentences2=[val[4] for val in dataset]
        self.input_sentences3=[val[5] for val in dataset]
        self.input_embeddings1 = [val[0][0] for val in dataset]
        self.input_embeddings2 = [val[1][0] for val in dataset]
        self.input_embeddings3 = [val[2][0] for val in dataset]

    def __len__(self):
        return len(self.input_embeddings1)

    def __getitem__(self, index):
        emb1 = hp.to_tensor(self.input_embeddings1[index])
        emb2 = hp.to_tensor(self.input_embeddings2[index])
        emb3 = hp.to_tensor(self.input_embeddings3[index])
        sent1=self.input_sentences1[index]
        sent2=self.input_sentences2[index]
        sent3=self.input_sentences3[index]

        return emb1, emb2, emb3, sent1, sent2, sent3

class CustomDataset(Dataset):
    def __init__(self, dataset,device):
        self.dataset=np.array(dataset,dtype=object)
        self.device = device
    def __len__(self):
        return len(self.dataset)

    def total_indexes(self):
        return np.unique(self.dataset[:, 3])

    def get_row_indexes(self,target_sample_index):
        return np.where(self.dataset[:, 3] == target_sample_index)[0]

    def get_samples_at_data_index(self,target_sample_index):
        row_indexes = np.where(self.dataset[:, 3] == target_sample_index)[0]
        emb1=[]
        emb2=[]
        label=[]
        row_index=[]
        sent1=[]
        sent2=[]
        for index in row_indexes:
        
          emb1.append(hp.to_tensor(self.dataset[index][0][0]))
          emb2.append(hp.to_tensor(self.dataset[index][1][0]))
          label.append(hp.to_tensor(self.dataset[index][2]))
          row_index.append(hp.to_tensor(self.dataset[index][3]))
          sent1.append(self.dataset[index][4])
          sent2.append(self.dataset[index][5])
        return emb1.to(self.device), emb2.to(self.device), label.to(self.device),row_index, sent1, sent2

    def __getitem__(self, index):
        emb1 = hp.to_tensor(self.dataset[index][0]).to(self.device)#, dtype=torch.float)
        emb2 = hp.to_tensor(self.dataset[index][1]).to(self.device)#, dtype=torch.float)
        label = hp.to_tensor(self.dataset[index][2]).to(self.device)#, dtype=torch.long)
        sample_index=self.dataset[index][3]
        sent1=self.dataset[index][4]
        sent2=self.dataset[index][5]


        return emb1, emb2, label,sample_index, sent1, sent2
    
def get_data_loader(dataset_paired,batch_size=2,shuffle=True,device="cpu"):
  """
    dataset: dataset to be used
    shuffle: dataset shuffle per iteration

  """

  dataset_pt=CustomDataset(dataset_paired,device)
  data_loader=DataLoader(dataset_pt, batch_size=batch_size, shuffle=shuffle)
  return data_loader

# DATA FORMATING FOR COUNTERFACT
    
def create_dataset_tripletloss(dataset,mode=0):
  """
  Modes:
    0 high sim as train and low sim as test
    1 low sim as test and high sim as test
    2 random assigment
  """
  dataset_paired_train=[]
  dataset_paired_test=[]
  for row in dataset:
    if(mode==0):
      for index,vector in enumerate(row["vectors_neighborhood_prompts_high_sim"]):

        dataset_paired_train.append([vector,row["vector_edited_prompt"],row["vector_edited_prompt_paraphrases_processed"],
                                     row["edited_prompt"][0],row["neighborhood_prompts_high_sim"][index],row["edited_prompt_paraphrases_processed"]])
      for index,vector in enumerate(row["vectors_neighborhood_prompts_low_sim"]):
        dataset_paired_test.append([vector,row["vector_edited_prompt"],row["vector_edited_prompt_paraphrases_processed_testing"],
                                   row["edited_prompt"][0], row["neighborhood_prompts_low_sim"][index],row["edited_prompt_paraphrases_processed_testing"]])
    elif(mode==1):
      for index,vector in enumerate(row["vectors_neighborhood_prompts_high_sim"]):
        dataset_paired_test.append([vector,row["vector_edited_prompt"],row["vector_edited_prompt_paraphrases_processed"],
                                    row["edited_prompt"][0],row["neighborhood_prompts_high_sim"][index],row["edited_prompt_paraphrases_processed_testing"]])
      for index,vector in enumerate(row["vectors_neighborhood_prompts_low_sim"]):
        dataset_paired_train.append([vector,row["vector_edited_prompt"],row["vector_edited_prompt_paraphrases_processed_testing"],
                                     row["edited_prompt"][0],row["neighborhood_prompts_low_sim"][index],row["edited_prompt_paraphrases_processed"]])
    else:
      chosen_elements_train = random.sample([ i for i in range(10)], k=5)
      for index,vector in enumerate(row["vectors_neighborhood_prompts_high_sim"]):
        if(index in chosen_elements_train):
          dataset_paired_train.append([vector,row["vector_edited_prompt"],row["vector_edited_prompt_paraphrases_processed"],
                                       row["edited_prompt"][0],row["neighborhood_prompts_high_sim"][index],row["edited_prompt_paraphrases_processed"]])
        else:
          dataset_paired_test.append([vector,row["vector_edited_prompt"],row["vector_edited_prompt_paraphrases_processed_testing"],
                                      row["edited_prompt"][0],row["neighborhood_prompts_high_sim"][index],row["edited_prompt_paraphrases_processed_testing"]])
      for index,vector in enumerate(row["vectors_neighborhood_prompts_low_sim"]):
        if((5+index) in chosen_elements_train):
          dataset_paired_train.append([vector,row["vector_edited_prompt"],row["vector_edited_prompt_paraphrases_processed"],
                                       row["edited_prompt"][0],row["neighborhood_prompts_low_sim"][index],row["edited_prompt_paraphrases_processed"]])
        else:
          dataset_paired_test.append([vector,row["vector_edited_prompt"],row["vector_edited_prompt_paraphrases_processed_testing"],
                                      row["edited_prompt"][0],row["neighborhood_prompts_low_sim"][index],row["edited_prompt_paraphrases_processed_testing"]])

  return  dataset_paired_train,dataset_paired_test


def create_dataset_pairs(dataset,neightbour_control=0,label_reversal=False):
  """
  Modes:
    0 high sim as train and low sim as test
    1 low sim as test and high sim as test
    2 random assigment
  """

  if(label_reversal==True):
    paraphrase=0
    neightbour=1
  else:
    paraphrase=1
    neightbour=0

  dataset_paired_train=[]
  dataset_paired_test=[]
  for row_index,row in enumerate(dataset):
    dataset_paired_train.append([row["vector_edited_prompt"],row["vector_edited_prompt_paraphrases_processed"],paraphrase,row_index,
                                 row["edited_prompt"][0],row["edited_prompt_paraphrases_processed"]])
    dataset_paired_test.append([row["vector_edited_prompt"],row["vector_edited_prompt_paraphrases_processed_testing"],paraphrase,row_index,
                                row["edited_prompt"][0],row["edited_prompt_paraphrases_processed_testing"]])
    if(neightbour_control==0):
      for index,vector in enumerate(row["vectors_neighborhood_prompts_high_sim"]):
        dataset_paired_train.append([vector,row["vector_edited_prompt"],neightbour,row_index,
                                     row["edited_prompt"][0],row["neighborhood_prompts_high_sim"][index]])
      for index,vector in enumerate(row["vectors_neighborhood_prompts_low_sim"]):
        dataset_paired_test.append([vector,row["vector_edited_prompt"],neightbour,row_index,
                                    row["edited_prompt"][0],row["neighborhood_prompts_low_sim"][index]])
    elif(neightbour_control==1):
      for index,vector in enumerate(row["vectors_neighborhood_prompts_high_sim"]):
        dataset_paired_test.append([vector,row["vector_edited_prompt"],neightbour,row_index,
                                    row["edited_prompt"][0],row["neighborhood_prompts_high_sim"][index]])
      for index,vector in enumerate(row["vectors_neighborhood_prompts_low_sim"]):
        dataset_paired_train.append([vector,row["vector_edited_prompt"],neightbour,row_index,
                                     row["edited_prompt"][0],row["neighborhood_prompts_low_sim"][index]])
    else:

      chosen_elements_train = random.sample([ i for i in range(10)], k=5)

      for index,vector in enumerate(row["vectors_neighborhood_prompts_high_sim"]):
        if(index in chosen_elements_train):
          dataset_paired_train.append([vector,row["vector_edited_prompt"],neightbour,row_index,
                                       row["edited_prompt"][0],row["neighborhood_prompts_high_sim"][index]])
        else:
          dataset_paired_test.append([vector,row["vector_edited_prompt"],neightbour,row_index,
                                      row["edited_prompt"][0],row["neighborhood_prompts_high_sim"][index]])

      for index,vector in enumerate(row["vectors_neighborhood_prompts_low_sim"]):
        if((5+index) in chosen_elements_train):
          dataset_paired_train.append([vector,row["vector_edited_prompt"],neightbour,row_index,
                                       row["edited_prompt"][0],row["neighborhood_prompts_low_sim"][index]])
        else:
          dataset_paired_test.append([vector,row["vector_edited_prompt"],neightbour,row_index,
                                      row["edited_prompt"][0],row["neighborhood_prompts_low_sim"][index]])

  return  dataset_paired_train,dataset_paired_test


def data_construct_high_sim(dataset,neightbour_control=0,label_reversal=False,comparison="dist",topk_neg=10,topk_pos=5):
  if(label_reversal==True):
    paraphrase=0
    neighbour=1
  else:
    paraphrase=1
    neighbour=0

  data_loader=get_data_loader(dataset,batch_size=1,shuffle=False)
  dataset_processed=[]
  vector_list_edits=[]
  vector_list_neighbours=[]
  neighbours_prompt=[]
  edit_prompts=[]
  row_indexes=[]
  for sample in data_loader:
    if(sample[2].item()==paraphrase):
      vector_list_edits.append(sample[0][0])
      edit_prompts.append(sample[4][0])
      row_indexes.append(sample[3][0].item())
    else:
      vector_list_neighbours.append(sample[0][0])
      neighbours_prompt.append(sample[5][0])
  vectors = torch.stack(vector_list_neighbours)
  for index_vector,target_vector in enumerate(vector_list_edits):
      metric = util.cos_sim(target_vector,vectors)
      top_indices = torch.topk(metric, k=topk_neg).indices
      for index in top_indices[0].numpy().tolist():
        dataset_processed.append([vector_list_neighbours[index],target_vector,neighbour,row_indexes[index_vector],
                                      edit_prompts[index_vector],neighbours_prompt[index]])

  return dataset_processed



# def data_construct_high_sim(dataset_indexed,matching_indexes,neightbour_control=0,reverse_label=False):
#   if(reverse_label==True):
#     praphrase=0
#     neighbour=1
#   else:
#     praphrase=1
#     neighbour=0

#   dataset_processed=[]
#   for key in matching_indexes:
#     if(len(matching_indexes[key])==0):
#       continue
#     for index in matching_indexes[key]:
#       sample1=dataset_indexed[key]
#       sample2=dataset_indexed[index]
#       dataset_processed.append([sample1["vector_edited_prompt"],sample2["vector_edited_prompt"],neighbour,key,
#                                   sample1["edited_prompt"][0],sample2["edited_prompt"][0]])#move positive away


#       if(neightbour_control==0):
#         for index,vector in enumerate(sample2["vectors_neighborhood_prompts_high_sim"]):
#           dataset_processed.append([vector,sample1["vector_edited_prompt"],neighbour,key,
#                                       sample1["edited_prompt"][0],sample2["neighborhood_prompts_high_sim"][index]])
#         for index,vector in enumerate(sample1["vectors_neighborhood_prompts_high_sim"]):
#           dataset_processed.append([vector,sample2["vector_edited_prompt"],neighbour,key,
#                                       sample2["edited_prompt"][0],sample1["neighborhood_prompts_high_sim"][index]])
#       elif(neightbour_control==1):
#         for index,vector in enumerate(sample2["vectors_neighborhood_prompts_low_sim"]):
#           dataset_processed.append([vector,sample1["vector_edited_prompt"],neighbour,key,
#                                       sample1["edited_prompt"][0],sample2["neighborhood_prompts_high_sim"][index]])
#         for index,vector in enumerate(sample1["vectors_neighborhood_prompts_low_sim"]):
#           dataset_processed.append([vector,sample2["vector_edited_prompt"],neighbour,key,
#                                       sample2["edited_prompt"][0],sample1["neighborhood_prompts_high_sim"][index]])

#   return dataset_processed

def create_dataset_pairs_projection(dataset,projector,neightbour_control=0,label_reversal=False):
  """
  Modes:
    0 high sim as train and low sim as test
    1 low sim as test and high sim as test
    2 random assigment
  """

  if(label_reversal):
    paraphrase=0
    neightbour=1
  else:
    paraphrase=1
    neightbour=0

  dataset_paired_train=[]
  dataset_paired_test=[]

  for row_index,row in enumerate(dataset):
    vector_edit_prompt=projector.transform(row["vector_edited_prompt"]).astype(np.float64)

    dataset_paired_train.append([vector_edit_prompt,projector.transform(row["vector_edited_prompt_paraphrases_processed"]),paraphrase,row_index,
                                 row["edited_prompt"][0],row["edited_prompt_paraphrases_processed"]])
    dataset_paired_test.append([vector_edit_prompt,projector.transform(row["vector_edited_prompt_paraphrases_processed_testing"]),paraphrase,row_index,
                                row["edited_prompt"][0],row["edited_prompt_paraphrases_processed_testing"]])
    if(neightbour_control==0):
      for index,vector in enumerate(row["vectors_neighborhood_prompts_high_sim"]):
        dataset_paired_train.append([projector.transform(vector).astype(np.float64),vector_edit_prompt,neightbour,row_index,
                                     row["edited_prompt"][0],row["neighborhood_prompts_high_sim"][index]])
      for index,vector in enumerate(row["vectors_neighborhood_prompts_low_sim"]):
        dataset_paired_test.append([projector.transform(vector).astype(np.float64),vector_edit_prompt,neightbour,row_index,
                                    row["edited_prompt"][0],row["neighborhood_prompts_low_sim"][index]])
    elif(neightbour_control==1):
      for index,vector in enumerate(row["vectors_neighborhood_prompts_high_sim"]):
        dataset_paired_test.append([projector.transform(vector).astype(np.float64),vector_edit_prompt,neightbour,row_index,
                                    row["edited_prompt"][0],row["neighborhood_prompts_high_sim"][index]])
      for index,vector in enumerate(row["vectors_neighborhood_prompts_low_sim"]):
        dataset_paired_train.append([projector.transform(vector).astype(np.float64),vector_edit_prompt,neightbour,row_index,
                                     row["edited_prompt"][0],row["neighborhood_prompts_low_sim"][index]])
    else:

      chosen_elements_train = random.sample([ i for i in range(10)], k=5)

      for index,vector in enumerate(row["vectors_neighborhood_prompts_high_sim"]):
        if(index in chosen_elements_train):
          dataset_paired_train.append([projector.transform(vector).astype(np.float64),vector_edit_prompt,neightbour,row_index,
                                       row["edited_prompt"][0],row["neighborhood_prompts_high_sim"][index]])
        else:
          dataset_paired_test.append([projector.transform(vector).astype(np.float64),row["vector_edited_prompt"],neightbour,row_index,
                                      row["edited_prompt"][0],row["neighborhood_prompts_high_sim"][index]])

      for index,vector in enumerate(row["vectors_neighborhood_prompts_low_sim"]):
        if((5+index) in chosen_elements_train):
          dataset_paired_train.append([projector.transform(vector).astype(np.float64),vector_edit_prompt,neightbour,row_index,
                                       row["edited_prompt"][0],row["neighborhood_prompts_low_sim"][index]])
        else:
          dataset_paired_test.append([projector.transform(vector).astype(np.float64),vector_edit_prompt,neightbour,row_index,
                                      row["edited_prompt"][0],row["neighborhood_prompts_low_sim"][index]])

  return  dataset_paired_train,dataset_paired_test