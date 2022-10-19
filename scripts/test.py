import torch
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def test_model(model, testloader):
    
    criterion = nn.CrossEntropyLoss()
    total_loss, total_acc, label_list, pred_list, output_list = test(model, testloader, criterion)
    
    return total_loss, total_acc, label_list, pred_list, output_list 


# Get the accuracy and predicted labels
def test(model, testloader, criterion):
    model.to("cpu")
    model.eval() 
    running_loss = 0.0
    running_corrects = 0
    label_list = []
    pred_list = []
    output_list = []

    for inputs, labels in testloader:
        input_size = len(testloader.dataset)
        with torch.no_grad():
            outputs = model(inputs)
            output_list.append(outputs[0][1])
            preds = torch.argmax(outputs, dim=1)
            labels = torch.argmax(labels, dim=1)
      
            loss = criterion(outputs, labels)
            pred_list.append(preds[0])
            label_list.append(labels[0])

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

    total_loss = running_loss / input_size
    total_acc = running_corrects.double() / input_size
    total_acc = total_acc.to("cpu").numpy()
    print('Loss: {:.4f} Acc: {:.4f}'.format(total_loss, total_acc))
    return total_loss, total_acc, label_list, pred_list, output_list

# Get the confusion matrix after getting test predictions
def get_conf_matrix(label_list, pred_list):

    conf_matrix_vit = confusion_matrix(y_true=label_list, y_pred=pred_list)
    cm_display = ConfusionMatrixDisplay( confusion_matrix = conf_matrix_vit, display_labels = ["No Leakage", "Leakage"])
#     print(conf_matrix_vit)
#     print("----------------------------------------------------------------")
#     print('ViT Precision: %.3f' % precision_score(label_list, pred_list))
#     print('ViT Recall: %.3f' % recall_score(label_list, pred_list))
#     print('ViT F1 Score: %.3f' % f1_score(label_list, pred_list))
    return conf_matrix_vit, cm_display