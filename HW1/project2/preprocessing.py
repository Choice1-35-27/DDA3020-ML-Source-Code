def print_basic_info_of_dataset(data):
    print('check the shape of dataset:', data.shape)  
    print('-----------------------------------------')
    print(data.info())
    print('-----------------------------------------')

    print('check NaN for features:')
    print(data.isnull().sum())
    print('-----------------------------------------')

def prepare_normalized_x_and_y(data):
    data = data.dropna(how='any')
    data = data.iloc[:, 2:]
    # features
    train_df = data.iloc[:, :-2]
    # target1: next day's max temperature
    y1_df = data.iloc[:, -2]
    # target2: next day's min temperature
    y2_df = data.iloc[:, -1]

    min_value = train_df.min()
    max_value = train_df.max()
    train_nor = (train_df - min_value) / (max_value - min_value)
    y1_df_nor = (y1_df - y1_df.min()) / (y1_df.max() - y1_df.min())
    y2_df_nor = (y2_df - y2_df.min()) / (y2_df.max() - y2_df.min())

    return train_nor, y1_df_nor, y2_df_nor


