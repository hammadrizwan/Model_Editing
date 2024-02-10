from imports import np, random, json,linecache
random.seed(42)
np.random.seed(42)# fix seed

def read_dataset(file_path_read_dataset: str,data_size):
    dataset=[]

    random_numbers=random.sample(range(1, 21000), data_size) 
    for number in random_numbers:
        # print(json.loads(linecache.getline(file_path_read_dataset, number)))
        # print(linecache.getline(file_path_read_dataset, number).strip())
        data_entry = json.loads(linecache.getline(file_path_read_dataset, number).strip())
        dataset.append(data_entry)
        
    return dataset

def read_dataset_reduced(file_path_read_dataset: str,data_size):
    dataset=[]
    values_list = list(range(1, data_size+1))
    for number in values_list:
        
        # print(json.loads(linecache.getline(file_path_read_dataset, number)))
        # print(linecache.getline(file_path_read_dataset, number).strip())
        data_entry = json.loads(linecache.getline(file_path_read_dataset, number).strip())
        dataset.append(data_entry)
        
    return dataset
def write_dataset(file_path_write_dataset: str,dataset):
    with open(file_path_write_dataset, 'w') as jsonl_file_writer:
        for row in dataset:
            json.dump(row, jsonl_file_writer)
            jsonl_file_writer.write('\n')

if __name__ == "__main__":
    num_samples=500
    print("Reading Data")
    file_path_read_dataset="/home/hrk21/projects/def-hsajjad/hrk21/datasets/counterface_dataset_avg_embedding.jsonl"
    
    dataset=read_dataset(file_path_read_dataset,data_size=num_samples)
    # print(dataset)
    
    print("Reading Data Completed")
    print("Writing Data")
    # print(dataset[0])
    file_path_write_dataset="/home/hrk21/projects/def-hsajjad/hrk21/datasets/counterface_dataset_avg_embedding_reduced_"+str(num_samples)+".jsonl"
    write_dataset(file_path_write_dataset,dataset)
    print("Writing Data Completed")
    print("Reread Dataset")
    dataset=read_dataset_reduced(file_path_write_dataset,data_size=num_samples)
    print(len(dataset))