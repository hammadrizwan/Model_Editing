from imports import json,os, random, np, torch, linecache
import trainer
import data_processing as dp
import models
import results_processing as rp
import helper_functions as hp
np.random.seed(42)# fux seed
random.seed(42)

def read_dataset_reduced(file_path_read_dataset: str,data_size):
    dataset=[]
    values_list = list(range(1, data_size+1))
    for number in values_list:
        
        # print(json.loads(linecache.getline(file_path_read_dataset, number)))
        # print(linecache.getline(file_path_read_dataset, number).strip())
        data_entry = json.loads(linecache.getline(file_path_read_dataset, number).strip())
        dataset.append(data_entry)
    return dataset

def read_dataset(file_path: str,data_size):
    dataset=[]
    random_numbers=random.sample(range(1, 21000), data_size) 
    for number in random_numbers:
        data_entry = json.loads(linecache.getline(file_path, number).strip())
        dataset.append(data_entry)
    return dataset
    # Open the file and read its content
    # with open(file_path, 'r') as file:
    #     # Read each line and parse it as JSON
    #     for line in file:
    #         data_row=json.loads(line.strip())
    #         dataset.append(data_row)

    # return random.sample(dataset, data_size)

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def generate_results(model,dataset_paired_train,dataset_paired_test,path_to_folder,mode,comparison,device):
    device="cpu"
    if(comparison=="dist"):
        initial_threhold=10
    else:
        initial_threhold=0.50
    data_loader_test_temp=dp.get_data_loader(dataset_paired_train,batch_size=1,shuffle=False,device=device)
    distances_sims,predictions_sim,threshold_predictions,targets=rp.get_model_results(model,data_loader_test_temp,threshold=initial_threhold,comparison=comparison,mode=mode)
    tablef=path_to_folder+'Similarit_Distance_Table_Train_'+loss+".png"
    hp.pretty_table_to_image(hp.get_sim_differences(distances_sims),output_file=tablef)

    output_file=path_to_folder+'ROC_PRC_Curves_'+loss+".png"
    precision, recall, thresholds_pr, _= hp.roc_prc_curves(targets,predictions_sim,output_file,comparison,False)

    if(comparison=="dist"):
        precision = np.array([1 if value == 0 else value for value in precision])
        recall = np.array([1 if value == 0 else value for value in recall])

    best_threshold=rp.get_best_threshold(precision, recall, thresholds_pr,predictions_sim,targets,comparison)
    distances_sims,predictions_sim,threshold_predictions,targets=rp.get_model_results(model,data_loader_test_temp,threshold=best_threshold,comparison=comparison,mode=mode)
    heat_map_file=path_to_folder+'ConfusionMatrix_Train_'+loss+".png"
    classification_report_file=path_to_folder+'Classificaiton_Report_Train_'+loss+".png"
    _=hp.write_classificaiton_report(targets,threshold_predictions,heat_map_file,classification_report_file)

    test_data_loader=dp.get_data_loader(dataset_paired_test,batch_size=1,shuffle=False,device=device)
    distances_sims,_,threshold_predictions,targets=rp.get_model_results(model,test_data_loader,threshold=best_threshold,comparison=comparison,mode=mode)
    tablef=path_to_folder+'Similarit_Distance_Table_Test_'+loss+".png"
    hp.pretty_table_to_image(hp.get_sim_differences(distances_sims),output_file=tablef)
    heat_map_file=path_to_folder+'ConfusionMatrix_Test'+loss+".png"
    classification_report_file=path_to_folder+'Classificaiton_Report_Test_'+loss+".png"
    _=hp.write_classificaiton_report(targets,threshold_predictions,heat_map_file,classification_report_file)
    #default results with single threshold
    file_path=path_to_folder+"/results_testset.jsonl"
    rp.write_result_to_file(model,dataset_paired_test,comparison,mode,file_path,device)
    file_path=path_to_folder+"/results_trainset.jsonl"
    rp.write_result_to_file(model,dataset_paired_train,comparison,mode,file_path,device)
    # check if the right sentence is matched on top result
    file_path=file_path=path_to_folder+"/results_vectorlist_matching_testset.jsonl"
    rp.write_results_to_file_vector_list(dataset_paired_train,dataset_paired_test,model,comparison,mode,file_path,device)

    # comparison="sim"
    # single threshold based results
    file_path=file_path=path_to_folder+"/results_vectorlist_indivisual_threshold_testset.jsonl"
    predictions,targets=rp.write_results_indivisual_threshold(dataset_paired_train,dataset_paired_test,model,comparison,mode,file_path,device)
    print(len(predictions),len(targets))
    heat_map_file=path_to_folder+'ConfusionMatrix_Test_Indivisual_Threshold_'+loss+".png"
    classification_report_file=path_to_folder+'Classificaiton_Report_Test_Indivisual_Threshold_'+loss+".png"
    _=hp.write_classificaiton_report(targets,predictions,heat_map_file,classification_report_file)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device",device)
    """
    if(label_reversal):
        paraphrase=0
        neightbour=1
    else:
        paraphrase=1
        neightbour=0
    """
    num_samples=500
    # file_path_dataset="/home/hrk21/projects/def-hsajjad/hrk21/datasets/counterface_dataset_avg_embedding.jsonl"
    file_path_dataset="/home/hrk21/projects/def-hsajjad/hrk21/datasets/counterface_dataset_avg_embedding_reduced_500.jsonl"
    control=0#version of datasplit to be used, 0 based on high sim, 1 on low sim and 3 on random 
    label_reversal=False# no longer required cost functions updated
    loss="cosine_crossentropy" # cosine, cosine_crossentropy, contrastive
    print("Loading data")
    # dataset=read_dataset(file_path_dataset,data_size=num_samples)
    dataset=read_dataset_reduced(file_path_dataset,data_size=num_samples) 
    print("Loading data completed")
    # Contrastive_Learning_Automated
    path_to_folder="./Contrastive_Learning_Automated_AVG_EMB/"+loss+"_mode_"+str(num_samples)+"_"+str(control)+"/"#+"sim_groupings"#+"projector_gaussian"#

    create_folder(path_to_folder)
    # dp.create_dataset_pairs(
    # dataset_paired_train,dataset_paired_test=    create_dataset_pairs_projection(dataset,projector,neightbour_control=0,label_reversal=False)
    print("Data Processing")
    dataset_paired_train,dataset_paired_test=dp.create_dataset_pairs(dataset,control,label_reversal)#neighbourhood selection type 0 and reverse labels for Constrastive
    dataset_paired_train=dataset_paired_train+dp.data_construct_high_sim(dataset_paired_train,neightbour_control=0,label_reversal=label_reversal,comparison="sim",topk_neg=20,topk_pos=0)
    print("Data Processing completed")
    print("Total Samples",len(dataset_paired_train))

    num_epochs=50
    early_stop_patience=7
    print("Training Start")
    model,mode,comparison,file_path=trainer.train_control(dataset_paired_train=dataset_paired_train,loss_function=loss,
            learning_rate=0.0001,weight_decay=0.001,path_to_folder=path_to_folder,epochs=num_epochs,cls_weights=False,device=device,early_stop_patience=early_stop_patience)#cosine, cosine_crossentropy, contrastive
    path_to_folder="./Contrastive_Learning_Automated_AVG_EMB/"+loss+"_mode_"+str(num_samples)+"_"+str(control)+"/"#+"sim_groupings"#+"projector_gaussian"#
    # model=models.SiameseClassificationNetwork(512)
    # state_dict = torch.load("/home/hrk21/projects/def-hsajjad/hrk21/datasets/Projector_Networks/Contrastive_Learning_Automated_AVG_EMB/cosine_crossentropy_mode_0best_model_weights.pth")
    # model.load_state_dict(state_dict)
    # comparison="sim"
    # mode="classificaiton"
    print(file_path)
    print("Training Completed")
    print("Result Generation Start")
    generate_results(model,dataset_paired_train,dataset_paired_test,path_to_folder,mode,comparison,device)
    print("Result Generation Completed")