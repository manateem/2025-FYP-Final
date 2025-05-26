import joblib
import os
def saveModel(model, modelName, directory):
    if ".pkl" not in modelName[-4:]:
        modelName += ".pkl"
    filepath = os.path.join(directory, modelName)
    if not os.path.exists(directory):
        os.makedirs(directory)
    joblib.dump(model, filepath)


