from imports import nn,torch,F

class CosineSimilarityLoss_SentenceTransformers(nn.Module):
  def __init__(self):
        super(CosineSimilarityLoss_SentenceTransformers, self).__init__()
        self.loss_fct = torch.nn.MSELoss()
        self.cos_score_transformation=nn.Identity()

  def forward(self, output1, output2, target):
      score= torch.cosine_similarity(output1.to(dtype=torch.float), output2.to(dtype=torch.float))
      score = self.cos_score_transformation(score)
      loss=self.loss_fct(score, target.to(dtype=torch.float).view(-1))
      return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - target) * 0.5 * torch.pow(euclidean_distance, 2) +
                                      (target) * 0.5 * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class WeightedContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, positive_weight=1.0, negative_weight=1.0):
        super(WeightedContrastiveLoss, self).__init__()
        self.margin = margin
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Weighted contributions from positive and negative samples
        loss_contrastive = torch.mean(0.5*(
            label * self.positive_weight  * torch.pow(euclidean_distance, 2) +
            (1 - label) * self.negative_weight  * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        ))
        
        return loss_contrastive






  # class ContrastiveLoss(nn.Module):
  #   """
  #   Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
  #   two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.

  #   Further information: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

  #   :param model: SentenceTransformer model
  #   :param distance_metric: Function that returns a distance between two embeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
  #   :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
  #   :param size_average: Average by the size of the mini-batch.

  #   """

  #   def __init__(
  #       self,
  #       model: SentenceTransformer,
  #       distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
  #       margin: float = 0.5,
  #       size_average: bool = True,
  #   ):
  #       super(ContrastiveLoss, self).__init__()
  #       self.distance_metric = distance_metric
  #       self.margin = margin
  #       self.model = model
  #       self.size_average = size_average

  #   def get_config_dict(self):
  #       distance_metric_name = self.distance_metric.__name__
  #       for name, value in vars(SiameseDistanceMetric).items():
  #           if value == self.distance_metric:
  #               distance_metric_name = "SiameseDistanceMetric.{}".format(name)
  #               break

  #       return {"distance_metric": distance_metric_name, "margin": self.margin, "size_average": self.size_average}

  #   def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
  #       reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
  #       assert len(reps) == 2
  #       rep_anchor, rep_other = reps
  #       distances = self.distance_metric(rep_anchor, rep_other)
  #       losses = 0.5 * (
  #           labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2)
  #       )
  #       return losses.mean() if self.size_average else losses.sum()