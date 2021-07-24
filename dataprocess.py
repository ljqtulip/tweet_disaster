import re
import pandas as pd

data_path = './data/'
train = pd.read_csv(data_path+'train.csv',index_col=0)
test = pd.read_csv(data_path+'test.csv',index_col=0)

def data_process(data):
    for index, row in data.iterrows():
        pat = '\'s|\'\'|\'t|#|-|\?|\!|\.|:|~|,|\'ll|\'m'
        pat1 = '\.\.+'
        pat2 = 'http://t.co/[0-9a-zA-Z_]+'
        text = row['text']
        text = re.sub(pat1, '~', text)
        text = re.sub(pat2, '', text)

        def blank(matched):
            result = matched.group()
            result = ' ' + result + ' '
            return result

        text = re.sub(pat, blank, text)
        text = re.sub('~', '...', text)
        text = re.sub('\s+',' ',text)
        data.at[index, 'text'] = text

data_process(train)
data_process(test)

train.to_csv(data_path+'train_process.csv')
test.to_csv(data_path+'test_process.csv')