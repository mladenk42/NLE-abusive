import pandas as pd
import os


dir_name = '../results/classify/'
all_dirs = [x[0] for x in os.walk(dir_name)]
all_dirs.sort()
result = pd.DataFrame()

for model_dir in all_dirs:
    file_name = model_dir +'/metrics.csv'
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df.val_loss = -1 * df.val_loss
        idx_loss = df.val_loss.idxmax()
        test_f1_loss = df.test_f1[idx_loss] * 100

        idx_f1 = df.val_f1.idxmax()
        test_f1_f1 = df.test_f1[idx_f1]  * 100

        idx_best = df.test_f1.idxmax()
        test_best_f1 = df.test_f1[idx_best]  * 100
        
        model_card = model_dir.replace(dir_name, '')
        data = {'model':model_card,'epoch_loss':idx_loss+1,'test_f1_loss':test_f1_loss,'epoch_f1':idx_f1+1,'test_f1_f1':test_f1_f1,'epoch_best':idx_best+1,'test_best_f1':test_best_f1}
        result = result.append(data, ignore_index=True)
        # print(model_dir)
print(result)

loss_file_name = dir_name + 'all_best_metrics.csv'
result.to_csv(loss_file_name, index=False)
print('Results saved to ', loss_file_name)
