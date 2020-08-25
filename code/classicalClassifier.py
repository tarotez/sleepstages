from sklearn import ensemble, linear_model, svm, neural_network
# import xgboost as xgb
from staticDecisionTreeClassifier import StaticDecisionTreeClassifier

class ClassicalClassifier():

  def __init__(self, params, paramID):
      self.classifierType = params.classifierType
      self.classifierParams = params.classifierParams
      self.paramID = paramID
      if self.classifierType == 'logreg':
          self.model =  linear_model.LogisticRegression(C=self.classifierParams[self.paramID])
      elif self.classifierType == 'svm':
          # self.model =  svm.SVC(kernel='rbf', C=params.classifierParams[paramID])
          self.model =  svm.SVC(kernel='rbf')
      elif self.classifierType == 'linear_svm':
          self.model =  svm.LinearSVC()
      elif self.classifierType == 'rf':
          # self.model =  ensemble.RandomForestClassifier(n_estimators=classifierParams[paramID], class_weight="balanced")
          self.model =  ensemble.RandomForestClassifier(n_estimators=self.classifierParams[self.paramID])
      elif self.classifierType == 'adaboost':
          self.model = ensemble.AdaBoostClassifier(n_estimators=self.classifierParams[self.paramID])
      elif self.classifierType == 'nn':
          self.model =  neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
      elif self.classifierType == 'static':
          self.model =  StaticDecisionTreeClassifier()
      elif self.classifierType == 'xgb':
          self.xgb_params = {}

  def train(self, x, y):
    #  if self.classifierType == 'xgb':
    #      dmat = xgb.DMatrix(x, label=y)
    #      self.model = xgb.train(self.xgb_params, dmat)
    #  else:
          sampleNum = len(y)
          x_flattened = x.reshape(sampleNum, -1)
          self.featureNum = x_flattened.shape[1]
          # print('x_flattened.shape =', x_flattened.shape)
          # print('x_flattened =', x_flattened)
          # print('y =', y)
          self.model.fit(x_flattened, y)

  def predict(self, x):
      x_flattened = x.reshape(-1, self.featureNum)
      # print('x_flattened =', x_flattened)
      return self.model.predict(x_flattened)
