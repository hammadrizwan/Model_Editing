from imports import (torch,util,ImageDraw,ImageFont,Image,plt,sns,PrettyTable,
                     classification_report,confusion_matrix,compute_sample_weight,
                     matplotlib,auc,roc_curve,average_precision_score,precision_recall_curve,
                     np,precision_score,recall_score,tqdm)

def to_tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x
    
def find_nearest_vector(target_tensor, vector_list,comparison):
    # Convert the target tensor to a PyTorch tensor
    # target_tensor = torch.tensor(target_tensor)

    # Convert the list of vectors to a PyTorch tensor
    vectors = torch.stack(vector_list)

    # # Calculate the Euclidean distances
    if(comparison=="dist"):
      metric = torch.norm(target_tensor - vectors, dim=1)
      best_distance_index = torch.argmin(metric).item()
      best_distance = torch.min(metric)
    else:
      #calculate cosine sim
      metric = util.cos_sim(target_tensor,vectors)
      best_distance_index = torch.argmax(metric).item()
      best_distance = torch.max(metric)
    # Find the index of the vector with the minimum distance
    
    # Return the vector with the minimum distance and its index
    return vector_list[best_distance_index], best_distance_index,best_distance

# PROJECTION
def projector_fit(projector_transform,dataset):
  vector_list=[]
  for row in dataset:
    vector_list.append(row["vector_edited_prompt"][0])

    vector_list.append(row["vector_edited_prompt_paraphrases_processed"][0])

    vector_list.append(row["vector_edited_prompt_paraphrases_processed_testing"][0])

    for vector in row["vectors_neighborhood_prompts_high_sim"]:
      vector_list.append(vector[0])
    for vector in row["vectors_neighborhood_prompts_low_sim"]:
      vector_list.append(vector[0])
  return projector_transform.fit(vector_list)

def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

#write to table
def pretty_table_to_image(table, font_size=22, cell_padding=10, output_file="table_image.png"):
    # Convert PrettyTable to string
    table_str = str(table)

    # Calculate the size of the image based on the text size
    font = ImageFont.load_default()  # Use default font
    # text_width, text_height = ImageDraw.Draw(Image.new("RGB", (1, 1))).multiline_textsize(table_str, font=font)
    text_width, text_height = textsize(table_str, font)
    # text_width = 570 
    # text_height= 86
    img_width = text_width + cell_padding * 2
    img_height = text_height + cell_padding * 2

    # Create a blank image
    img = Image.new("RGB", (img_width, img_height), color="white")
    draw = ImageDraw.Draw(img)

    # Draw the table on the image
    draw.multiline_text((cell_padding, cell_padding), table_str, font=font, fill="black")

    # Save the image
    img.save(output_file)


#classification report
    
def write_classificaiton_report(targets,predictions,output_file_heatmap,output_file_confusion_matrix):
  report = classification_report(targets, predictions)
  cm = confusion_matrix(targets, predictions)

  # Plot the confusion matrix as a heatmap using Seaborn
  plt.figure(figsize=(4, 4))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
              xticklabels=["Class 0", "Class 1"],
              yticklabels=["Class 0", "Class 1"])
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.title("Confusion Matrix")
  plt.show()
  plt.savefig(output_file_heatmap)

  # Convert classification report to PrettyTable
  # Check if the report is a dictionary or a string
  if isinstance(report, dict):
      # Convert classification report to PrettyTable
      table = PrettyTable()
      table.field_names = ["Class"] + list(report.keys())
      for label, values in report.items():
          if label not in ['accuracy', 'macro avg', 'weighted avg']:
              table.add_row([label] + [values[key] for key in table.field_names[1:]])

      # Convert PrettyTable to string
      table_str = str(table)
  else:
      # If the report is already a string, use it directly
      table_str = report

  # Calculate the size of the image based on the text size
  font = ImageFont.load_default()  # Use default font
  # text_width , text_height = ImageDraw.Draw(Image.new("RGB", (1, 1))).multiline_textsize(table_str, font=font)
  text_width, text_height = textsize(table_str, font)
  img_width = text_width + 20  # Add padding
  img_height = text_height + 20  # Add padding

  # Create a blank image
  img = Image.new("RGB", (img_width, img_height), color="white")
  draw = ImageDraw.Draw(img)

  # Draw the table on the image
  draw.multiline_text((10, 10), table_str, font=font, fill="black")


  # Save the image
  img.save(output_file_confusion_matrix)

  return cm


#ROC CURVES
def roc_prc_curves(targets,predictions,output_file,comparison,avg_macro=False):
  matplotlib.use('Agg')  # Use the Agg backend
  fpr, tpr, thresholds = roc_curve(targets, predictions)
  # if(comparison=="dist"):
  #   fpr, tpr, thresholds = roc_curve(targets, predictions,pos_label=1)
  # else:
  #   fpr, tpr, thresholds = roc_curve(targets, predictions,pos_label=0)

  # Calculate the area under the ROC curve (AUC)
  roc_auc = auc(fpr, tpr)
  sample_weights = compute_sample_weight(class_weight='balanced', y=targets)
  # print(sample_weights)
  precision, recall, thresholds_pr = precision_recall_curve(targets, predictions,sample_weight=sample_weights)
  average_precision = average_precision_score(targets, predictions,average="macro")
  
  if(avg_macro):
    precision=[]
    recall=[]
    for threshold in thresholds_pr:
      y_pred = [1 if value < threshold else 0 for value in predictions]
      precision.append(precision_score(targets, y_pred, average='macro'))
      recall.append(recall_score(targets, y_pred, average='macro'))
    recall=np.array(recall)
    precision=np.array(precision)
  # Plot the ROC curve
  plt.figure(figsize=(12, 5))

  plt.subplot(1, 2, 1)
  plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
  plt.xlabel('False Positive Rate (FPR)')
  plt.ylabel('True Positive Rate (TPR)')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend(loc='lower right')

  # Plot the Precision-Recall curve
  plt.subplot(1, 2, 2)
  plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall Curve')
  plt.legend(loc='upper right')

  plt.tight_layout()
  plt.show()

  plt.savefig(output_file)
  return precision, recall, thresholds_pr, average_precision

#similarity difference table
def get_sim_differences(distances_sims):
  # Convert the list to a NumPy array
  arr = np.array(distances_sims)
  # Extract labels and values
  labels = arr[:, 2].astype(int)
  # print(values)
  values = arr[:, :2]

  # Create a dictionary to store sums and counts for each label
  sums = {label: np.zeros(2) for label in np.unique(labels)}
  counts = {label: 0 for label in np.unique(labels)}

  # Calculate sums and counts for each label
  for label, value in zip(labels, values):
      sums[label] += value
      counts[label] += 1

  # Calculate averages for each label
  averages = {label: sums[label] / counts[label] for label in sums}

  # Calculate change in similarity for each label
  change_in_similarity = {label: abs(averages[label][1] - averages[label][0]) for label in averages}

  table = PrettyTable()
  table.field_names = ["Label", "Average Sim After Projection", "Average Sim Before Projection", "Change in Similarity"]

  # Populate the table
  for label in averages:
      table.add_row([label, averages[label][0], averages[label][1], change_in_similarity[label]])

  return table



def threshold_per_index(model,dataset_pt,targets,predictions_sim,comparison,mode):
  predictions_sim=np.array(predictions_sim)
  targets=np.array(targets)
  
  sample_indexes=dataset_pt.total_indexes()
  vector_list=[]
  threshold_list=[]
  string_list=[]
  for sample_index in tqdm(sample_indexes, desc="Creating thresholds", unit="batch"):
    # print("sample_index",sample_index)
    example_indexes=dataset_pt.get_row_indexes(sample_index)#gry 
    # print(example_indexes)
    # print("example_indexes",len(example_indexes))
    otpimal_dist=find_optimal_threshold(predictions_sim[example_indexes],targets[example_indexes],comparison)
    with torch.no_grad():
      # print(dataset_pt[example_indexes[0]][1],dataset_pt[example_indexes[0]][0])
      emb1, emb2, _,_, _, _,_=dataset_pt.__getitem__(example_indexes[0])# zero is always the positive sample
      emb1=emb1.unsqueeze(0)
      emb2=emb2.unsqueeze(0)
      if(mode=="similarity"):
        output1, _ = model(emb1,emb2)
      else:
        output1, _, _ = model(emb1,emb2)
    
    vector_list.append(output1[0])
    string_list.append(dataset_pt[example_indexes[0]][4])
    threshold_list.append(otpimal_dist)
  return vector_list,threshold_list,string_list


def find_optimal_threshold(distances, ground_truth_labels, comparison="dist"):
    # Combine distances and labels into tuples
    data = list(zip(distances, ground_truth_labels))
    # Sort the data by distances in descending order
    # print(comparison)
    if(comparison=="dist"):
      sorted_data = sorted(data, key=lambda x: x[0], reverse=True)

      # Separate positive and negative class distances
      positive_distances = [dist for dist, label in sorted_data if label == 1]
      negative_distances = [dist for dist, label in sorted_data if label == 0]
      # print(positive_distances,negative_distances)
      positive_candidate = sum(positive_distances)/len(positive_distances)
      threshold_candidate_neg=10000
      for dist in negative_distances:
        if(dist>positive_candidate and dist<threshold_candidate_neg):
          threshold_candidate_neg=dist
      optimal_threshold = positive_candidate + 5
    else:
      sorted_data = sorted(data, key=lambda x: x[0], reverse=False)

      # Separate positive and negative class distances
      positive_distances = [dist for dist, label in sorted_data if label == 1]
      negative_distances = [dist for dist, label in sorted_data if label == 0]
      # print(positive_distances,negative_distances)
      positive_candidate = sum(positive_distances)/len(positive_distances)
      threshold_candidate_neg=-1
      for sim in negative_distances:
        if(sim<positive_candidate and sim>threshold_candidate_neg):
          threshold_candidate_neg=sim
    # Set the threshold between the highest positive value and the selected negative value
      optimal_threshold = (positive_candidate+threshold_candidate_neg)/2 # + threshold_candidate_neg) / 2)

    return optimal_threshold