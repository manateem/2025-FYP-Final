from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import GroupKFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from SaveModel import saveModel
from load_dataset import loadDataFrameWithFeatures
from constants import p

MODEL_DIR = p("test/")
MODEL_NAME = "modelKNN.pkl"

DF = loadDataFrameWithFeatures(write_csv_to="result/dataset.csv")

baseline_feats = [col for col in DF.columns if col.startswith("feat_")]
print(baseline_feats)
x_all = DF[baseline_feats]
y_all = DF["biopsed"]

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=42)
print(y_train)
gkf = GroupKFold(5)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Test Accuracy:", acc)
print("Confusion Matrix:\n", cm)
print(recall_score(y_test,y_pred))
# save the model
saveModel(model, MODEL_NAME, MODEL_DIR)