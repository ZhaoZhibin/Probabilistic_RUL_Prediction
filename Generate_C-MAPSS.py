import numpy as np
import pandas as pd
import os

data_dir = '../CMAPSSData/'
save_dir = './data/'

# DROPLIST ['S18', 'S1', 'S19', 'S5', 'S16', 'S10', 'C2', 'S6', 'C1', 'C3']
# ['S2', 'S3', 'S4', 'S7', 'S8', 'S9', 'S11', 'S12', 'S13', 'S14', 'S15', 'S17', 'S20', 'S21']
data_name = 'FD003'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def add_rul(df, total_rul, end_point):
    engine_number = df['ID'].max()
    for i in range(engine_number):
        index = df[df['ID'] == i + 1].index
        # whether the true RUL larger than assumed total RUL
        if end_point[i] > total_rul:
            df.loc[index, 'RUL'] = total_rul #end_point[i]
        else:
            rul = df.loc[index, 'Cycle'].sort_values(ascending=False) - 1 + end_point[i]
            rul[0: len(index) - total_rul + int(end_point[i])] = total_rul
            df.loc[index, 'RUL'] = list(rul)
    return df




def pre_processing(df, cols, type='z-score'):

    for j in range(len(cols)):
        if type == '-1-1':
            min_value = df.loc[:, cols[j]].min()
            max_value = df.loc[:, cols[j]].max()
            df.loc[:, cols[j]] = 2 * (df.loc[:, cols[j]] - min_value) / (max_value - min_value) - 1
        elif type == 'z-score':
            mean_value = df.loc[:, cols[j]].mean()
            std_value = df.loc[:, cols[j]].std()
            df.loc[:, cols[j]] = (df.loc[:, cols[j]] - mean_value) / std_value
        else:
            raise NameError('This normalization is not included!')
    return df







def pre_processing_combined(df_train, df_test, cols, type='z-score'):

    for j in range(len(cols)):
        if type == '-1-1':
            min_value = df_train.loc[:, cols[j]].min()
            max_value = df_train.loc[:, cols[j]].max()
            df_train.loc[:, cols[j]] = 2 * (df_train.loc[:, cols[j]] - min_value) / (max_value - min_value) - 1
            df_test.loc[:, cols[j]] = 2 * (df_test.loc[:, cols[j]] - min_value) / (max_value - min_value) - 1
        elif type == 'z-score':
            mean_value = df_train.loc[:, cols[j]].mean()
            std_value = df_train.loc[:, cols[j]].std()
            df_train.loc[:, cols[j]] = (df_train.loc[:, cols[j]] - mean_value) / std_value
            df_test.loc[:, cols[j]] = (df_test.loc[:, cols[j]] - mean_value) / std_value
        else:
            raise NameError('This normalization is not included!')
    return df_train, df_test



def prepare_data(df, win_len, cols):
    labels = []
    engine_id = []
    cycles = []
    data = []
    engine_number = df['ID'].max()
    for i in range(engine_number):
        index = df[df['ID'] == i + 1].index
        # print(index)
        # whether the length of cycle smaller than win_len, if smaller, we pad the closet point in the front
        k = len(index)
        if k < win_len:
            labels.append(df.loc[index[-1], 'RUL'])
            engine_id.append(i + 1)
            cycles.append(index[-1])
            data.append(np.hstack(np.ones((win_len-k, len(cols)))*df.loc[0, cols].values.reshape(1, -1),
                                  df.loc[:, cols].values.reshape(k, -1)))
        else:
            for j in range(k-win_len+1):
                labels.append(df.loc[index[win_len+j-1], 'RUL'])
                engine_id.append(i + 1)
                cycles.append(win_len+j)
                #print(df.loc[index[j:j+win_len], cols].values.shape)
                data.append(df.loc[index[j:j+win_len], cols].values)
    prepared_df = pd.DataFrame()
    prepared_df['engine_id'] = engine_id
    prepared_df['cycle'] = cycles
    prepared_df['data'] = data
    prepared_df['label'] = labels
    return prepared_df


def prepare_data_test(df, win_len, cols):
    labels = []
    engine_id = []
    cycles = []
    data = []
    engine_number = df['ID'].max()
    for i in range(engine_number):
        index = df[df['ID'] == i + 1].index
        # whether the length of cycle smaller than win_len, if smaller, we pad the closet point in the front
        k = len(index)
        if k < win_len:
            labels.append(df.loc[index[-1], 'RUL'])
            engine_id.append(i + 1)
            cycles.append(index[-1])
            data.append(np.hstack(np.ones((win_len-k, len(cols)))*df.loc[0, cols].values.reshape(1, -1),
                                  df.loc[:, cols].values.reshape(k, -1)))
        else:
            labels.append(df.loc[index[-1], 'RUL'])
            engine_id.append(i + 1)
            cycles.append(k)
            #print(df.loc[index[k-win_len:], cols].values.shape)
            data.append(df.loc[index[k-win_len:], cols].values)
    prepared_df = pd.DataFrame()
    prepared_df['engine_id'] = engine_id
    prepared_df['cycle'] = cycles
    prepared_df['data'] = data
    prepared_df['label'] = labels
    return prepared_df



# Read the data
data_train_path = data_dir + 'train_' + data_name + '.txt'
data_test_path = data_dir + 'test_' + data_name + '.txt'
data_rul_path = data_dir + 'RUL_' + data_name + '.txt'
test_rul = np.loadtxt(data_rul_path)
print(test_rul.shape)
data_train = pd.read_csv(data_train_path, delim_whitespace=True, header=None)
data_test = pd.read_csv(data_test_path, delim_whitespace=True, header=None)
new_cols = ['ID', 'Cycle', 'C1', 'C2', 'C3'] + ['S' + str(x) for x in range(1, 26 - 4)]
data_train.columns = new_cols
data_test.columns = new_cols

# ['S2', 'S3', 'S4', 'S7', 'S8', 'S9', 'S11', 'S12', 'S13', 'S14', 'S15', 'S17', 'S20', 'S21']
# Drop the useless columns
droplist = []
engine_number = data_train['ID'].max()
for i in range(engine_number):
    df = data_train[data_train['ID'] == i + 1]
    droplist = list(set(droplist + list(df.std()[(df.std() < 1e-3)].index)))

engine_number = data_test['ID'].max()
for i in range(engine_number):
    df = data_test[data_test['ID'] == i + 1]
    droplist = list(set(droplist + list(df.std()[(df.std() < 1e-3)].index)))

drop_list = list(set(droplist + ['C1', 'C2', 'C3']))
drop_list.remove('ID')

print(drop_list)
data_train = data_train.drop(drop_list, axis=1)
data_test = data_test.drop(drop_list, axis=1)


# Normalization
cols = [x for x in data_train.columns if 'S' in x]
print(cols)
print(len(cols))
#data_train = pre_processing(data_train, cols, type='-1-1')
#data_test = pre_processing(data_test, cols, type='-1-1')

data_train, data_test = pre_processing_combined(data_train, data_test, cols, type='-1-1')

# Add RUL
data_train = add_rul(data_train, total_rul=125, end_point=np.zeros(data_train['ID'].max()))
data_test = add_rul(data_test, total_rul=125, end_point=test_rul)

# Generate Training and testing examples
prepared_train = prepare_data(data_train, win_len=30, cols=cols)
prepared_test = prepare_data_test(data_test, win_len=30, cols=cols)
prepared_test_all = prepare_data(data_test, win_len=30, cols=cols)


#prepared_test.to_csv(save_dir + 'test_' + data_name + '.csv', index=0)
#prepared_train.to_csv(save_dir + 'train_' + data_name + '.csv', index=0)


# Read pd.read_pickle('test.pkl')
prepared_test_all.to_pickle(save_dir + 'test_all_' + data_name + '.pkl')
prepared_test.to_pickle(save_dir + 'test_' + data_name + '.pkl')
prepared_train.to_pickle(save_dir + 'train_' + data_name + '.pkl')