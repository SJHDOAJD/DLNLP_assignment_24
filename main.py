import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore")

from A.LSTM import preprocess_LSTM
from A.LSTM import model_LSTM
from A.LSTM import train_LSTM
from A.LSTM import evaluate_LSTM
from B.BiLSTM import preprocess_BiLSTM
from B.BiLSTM import model_BiLSTM
from B.BiLSTM import train_BiLSTM
from B.BiLSTM import evaluate_BiLSTM
from C.BERT import preprocess_BERT
from C.BERT import model_BERT
from C.BERT import train_BERT
from C.BERT import evaluate_BERT

# ======================================================================================================================
# LSTM Method

#train_loader1, valid_loader1, test_loader1, tokenizer = preprocess_LSTM()
#model1 = model_LSTM(tokenizer)
#acc_train1, acc_valid1, ls_train1, ls_valid1 = train_LSTM(model1, train_loader1, valid_loader1)
#test_acc1, test_loss1 = evaluate_LSTM(model1, test_loader1)

#print('ALL accuracy:{},{},{};ALL loss:{},{},{};'.format(acc_train1, acc_valid1, test_acc1, ls_train1, ls_valid1, test_loss1))

# ======================================================================================================================
# Bi-LSTM Method

train_loader2, valid_loader2, test_loader2, tokenizer2 = preprocess_BiLSTM()
model2 = model_BiLSTM(tokenizer2)
acc_train2, acc_valid2, ls_train2, ls_valid2 = train_BiLSTM(model2, train_loader2, valid_loader2)
test_acc2, test_loss2 = evaluate_BiLSTM(model2, test_loader2)

print('ALL accuracy:{},{},{};ALL loss:{},{},{};'.format(acc_train2, acc_valid2, test_acc2, ls_train2, ls_valid2, test_loss2))

# ======================================================================================================================
# BERT Method

#train_loader3, valid_loader3, test_loader3 = preprocess_BERT()
#model3 = model_BERT()
#acc_train3, acc_valid3, ls_train3, ls_valid3 = train_BERT(model3, train_loader3, valid_loader3)
#test_acc3, test_loss3 = evaluate_BERT(model3, test_loader3)

#print('ALL accuracy:{},{},{};ALL loss:{},{},{};'.format(acc_train3, acc_valid3, test_acc3, ls_train3, ls_valid3, test_loss3))

# ======================================================================================================================